//! Cross-area integration tests: the canonical examples from
//! [XPath and XQuery Functions and Operators 3.1](https://www.w3.org/TR/xpath-functions-31/),
//! exercised through the public API of `oxixml-fn` rather than through a module's own tests.

use oxixml_fn::{
    CASELESS_COLLATION, CollectingTrace, FnContext, FnErrorCode, FnResult, FunctionItem,
    HTML_ASCII_CASE_INSENSITIVE_COLLATION, accessors, arrays, boolean, datetime, diagnostics,
    higher_order, maps, nodes, numeric, qnames, registry, sequences, strings, xmlio,
};
use oxixml_infoset::{
    AtomicType, AtomicValue, DuplicateKeys, ExpandedName, Item, Sequence, Tree, XdmArray, XdmMap,
    XdmNode,
};
use oxixml_xsd::{DateTime, TimezoneOffset};
use std::str::FromStr;
use std::sync::Arc;

type AnyError = Box<dyn std::error::Error>;

fn parse(atomic_type: AtomicType, lexical: &str) -> AtomicValue {
    AtomicValue::parse(atomic_type, lexical).expect("a valid lexical form")
}

fn integers(values: &[i64]) -> Sequence {
    Sequence::from_iter(values.iter().copied().map(AtomicValue::integer))
}

fn rendered(sequence: &Sequence) -> String {
    sequence
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// A context pinned to the instant the specification's `fn:current-dateTime` example uses, so
/// that every test in this file is deterministic.
fn fixed_context() -> Result<FnContext, AnyError> {
    Ok(FnContext::new()
        .with_current_date_time(DateTime::from_str("2004-05-12T18:17:15.125Z")?)
        .with_implicit_timezone(TimezoneOffset::new(-300)?)
        .with_static_base_uri("http://example.com/base/"))
}

#[test]
fn accessors_over_a_parsed_document() -> Result<(), AnyError> {
    let tree = Tree::parse_str("<p:a xmlns:p='urn:p' xml:base='http://example.com/'>text</p:a>")?;
    let root = tree.document_element().ok_or("no document element")?;

    let name = accessors::node_name(&root).ok_or("an element has a name")?;
    assert_eq!(name.namespace(), Some("urn:p"));
    assert_eq!(name.local_name(), "a");
    assert_eq!(accessors::string(Some(&Item::from(root.clone())))?, "text");
    assert_eq!(accessors::data(&Sequence::from(root.clone()))?.len(), 1);
    assert_eq!(
        accessors::base_uri(&root).map(|value| value.string_value()),
        Some("http://example.com/".to_owned())
    );
    assert_eq!(accessors::nilled(&root), Some(false));
    assert_eq!(accessors::document_uri(&root), None);
    Ok(())
}

#[test]
fn numeric_operators_and_functions() -> Result<(), AnyError> {
    // op:numeric-* with promotion.
    assert_eq!(
        numeric::add(
            &AtomicValue::integer(1_i64),
            &parse(AtomicType::Decimal, "2.5")
        )?
        .string_value(),
        "3.5"
    );
    assert_eq!(
        numeric::divide(&AtomicValue::integer(3_i64), &AtomicValue::integer(2_i64))?.string_value(),
        "1.5"
    );
    assert_eq!(
        numeric::integer_divide(&AtomicValue::integer(-3_i64), &AtomicValue::integer(2_i64))?
            .string_value(),
        "-1"
    );
    assert!(
        numeric::divide(&AtomicValue::integer(1_i64), &AtomicValue::integer(0_i64))
            .expect_err("division by zero")
            .is(FnErrorCode::Foar0001)
    );

    // §4.4.
    assert_eq!(
        numeric::abs(&AtomicValue::integer(-10_i64))?.string_value(),
        "10"
    );
    assert_eq!(
        numeric::ceiling(&parse(AtomicType::Decimal, "-10.5"))?.string_value(),
        "-10"
    );
    assert_eq!(
        numeric::floor(&parse(AtomicType::Decimal, "-10.5"))?.string_value(),
        "-11"
    );
    assert_eq!(
        numeric::round(&parse(AtomicType::Decimal, "-2.5"), 0)?.string_value(),
        "-2"
    );
    assert_eq!(
        numeric::round_half_to_even(&parse(AtomicType::Decimal, "2.5"), 0)?.string_value(),
        "2"
    );

    // fn:number never raises.
    assert!(numeric::number(Some(&Item::from(AtomicValue::string("abc"))))?.is_nan());
    Ok(())
}

#[test]
fn format_number_picture_strings() -> Result<(), AnyError> {
    let context = FnContext::new();
    let format = context.default_decimal_format();
    let cases = [
        ("12345.6", "#,###.00", "12,345.60"),
        ("12345.6", "#,##0.00", "12,345.60"),
        ("0.234", "0.0000", "0.2340"),
        ("0.234", "#.00%", "23.40%"),
        ("0.234", "#.000%", "23.400%"),
    ];
    for (value, picture, expected) in cases {
        assert_eq!(
            numeric::format_number(Some(&parse(AtomicType::Decimal, value)), picture, format)?,
            expected,
            "format-number({value}, '{picture}')"
        );
    }
    assert_eq!(
        numeric::format_number(Some(&AtomicValue::integer(2500_i64)), "#,###.00", format)?,
        "2,500.00"
    );
    assert!(
        numeric::format_number(Some(&AtomicValue::integer(1_i64)), "0;0;0", format)
            .expect_err("two pattern separators")
            .is(FnErrorCode::Fodf1310)
    );
    Ok(())
}

#[test]
fn string_functions() -> Result<(), AnyError> {
    let context = FnContext::new();

    assert_eq!(
        strings::concat(&[
            Sequence::from(AtomicValue::string("un")),
            Sequence::from(AtomicValue::string("deux")),
        ])?,
        "undeux"
    );
    assert_eq!(
        strings::string_join(
            &Sequence::from_iter(["Now", "is", "the", "time"].map(AtomicValue::string)),
            " "
        )?,
        "Now is the time"
    );
    assert_eq!(strings::substring("motor car", 6.0, None), " car");
    assert_eq!(strings::substring("12345", 1.5, Some(2.6)), "234");
    assert_eq!(
        strings::string_length("Harp not on that string, madam; that is past."),
        45
    );
    assert_eq!(
        strings::normalize_space(" The    wealthy curled darlings\n of our nation. "),
        "The wealthy curled darlings of our nation."
    );
    assert_eq!(strings::translate("--aaa--", "abc-", "ABC"), "AAA");
    assert_eq!(
        strings::codepoints_to_string(&[84, 104, 232, 114, 232, 115, 101])?,
        "Thèrèse"
    );
    assert_eq!(
        strings::string_to_codepoints("Thérèse"),
        vec![84, 104, 233, 114, 232, 115, 101]
    );

    // Collation-sensitive comparison.
    assert_eq!(strings::compare(&context, "abc", "abc", None)?, 0);
    assert_eq!(strings::compare(&context, "Strasse", "Straße", None)?, -1);
    assert_eq!(
        strings::compare(
            &context,
            "ABC",
            "abc",
            Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION)
        )?,
        0
    );
    assert!(strings::contains(&context, "tattoo", "t", None)?);
    assert_eq!(
        strings::substring_before(&context, "tattoo", "attoo", None)?,
        "t"
    );
    assert_eq!(
        strings::substring_after(&context, "tattoo", "tat", None)?,
        "too"
    );
    assert!(
        strings::compare(&context, "a", "b", Some("urn:example:nope"))
            .expect_err("unknown collation")
            .is(FnErrorCode::Foch0002)
    );
    Ok(())
}

#[test]
fn regular_expression_functions() -> FnResult<()> {
    assert!(strings::matches("abracadabra", "^a.*a$", "")?);
    assert_eq!(strings::replace("abracadabra", "bra", "*", "")?, "a*cada*");
    assert_eq!(
        strings::replace("darted", "^(.*?)d(.*)$", "$1c$2", "")?,
        "carted"
    );
    assert_eq!(
        strings::tokenize("1, 15, 24, 50", ",\\s*", "")?,
        ["1", "15", "24", "50"]
    );
    assert!(
        strings::replace("abracadabra", ".*?", "$1", "")
            .expect_err("matches the empty string")
            .is(FnErrorCode::Forx0003)
    );
    Ok(())
}

#[test]
fn uri_functions() -> Result<(), AnyError> {
    let context = fixed_context()?;
    assert_eq!(strings::encode_for_uri("100% organic"), "100%25%20organic");
    assert_eq!(
        strings::iri_to_uri("http://www.example.com/~bébé"),
        "http://www.example.com/~b%C3%A9b%C3%A9"
    );
    assert_eq!(
        strings::escape_html_uri("http://www.example.com/00/Weather/CA/Los Angeles#ocean"),
        "http://www.example.com/00/Weather/CA/Los Angeles#ocean"
    );
    assert_eq!(
        strings::resolve_uri(&context, "sub/page.html")?,
        "http://example.com/base/sub/page.html"
    );
    Ok(())
}

#[test]
fn boolean_functions() -> Result<(), AnyError> {
    assert!(!boolean::boolean(&Sequence::empty())?);
    assert!(boolean::boolean(&Sequence::from(AtomicValue::string("x")))?);
    assert!(!boolean::boolean(&Sequence::from(AtomicValue::double(
        f64::NAN
    )))?);
    assert!(boolean::not(&Sequence::empty())?);
    assert!(
        boolean::boolean(&Sequence::from(parse(AtomicType::Date, "2020-01-01")))
            .expect_err("no effective boolean value")
            .is(FnErrorCode::Forg0006)
    );
    Ok(())
}

#[test]
fn date_time_components_and_arithmetic() -> Result<(), AnyError> {
    let context = fixed_context()?;
    let instant = parse(AtomicType::DateTime, "1999-05-31T13:20:00-05:00");
    assert_eq!(datetime::year_from_date_time(&instant)?, 1999);
    assert_eq!(datetime::month_from_date_time(&instant)?, 5);
    assert_eq!(datetime::hours_from_date_time(&instant)?, 13);
    assert_eq!(
        datetime::years_from_duration(&parse(AtomicType::YearMonthDuration, "P20Y15M"))?,
        21
    );

    // Arithmetic, including the operators of §8.4.
    assert_eq!(
        numeric::add(
            &parse(AtomicType::DateTime, "2000-10-30T11:12:00"),
            &parse(AtomicType::YearMonthDuration, "P1Y2M"),
        )?
        .string_value(),
        "2001-12-30T11:12:00"
    );
    assert_eq!(
        numeric::multiply(
            &parse(AtomicType::YearMonthDuration, "P2Y11M"),
            &AtomicValue::double(2.3),
        )?
        .string_value(),
        "P6Y9M"
    );

    // Timezone adjustment.
    assert_eq!(
        datetime::adjust_date_time_to_timezone(
            &context,
            &parse(AtomicType::DateTime, "2002-03-07T10:00:00"),
            datetime::TimezoneArgument::Implicit,
        )?
        .string_value(),
        "2002-03-07T10:00:00-05:00"
    );

    // The current instant is stable and comes from the context.
    assert_eq!(
        datetime::current_date_time(&context).string_value(),
        "2004-05-12T18:17:15.125Z"
    );
    assert_eq!(
        datetime::implicit_timezone(&context).string_value(),
        "-PT5H"
    );
    Ok(())
}

#[test]
fn date_time_picture_strings() -> Result<(), AnyError> {
    let context = FnContext::new();
    let instant = parse(AtomicType::DateTime, "2002-12-31T19:20:00-05:00");
    let cases = [
        (
            "[Y0001]-[M01]-[D01]T[H01]:[m01]:[s01][Z]",
            "2002-12-31T19:20:00-05:00",
        ),
        ("[D1] [MNn] [Y]", "31 December 2002"),
        ("[FNn], [D1o] [MNn] [Y]", "Tuesday, 31st December 2002"),
        ("[h]:[m01] [Pn]", "7:20 pm"),
        ("[Y01]", "02"),
        ("[MI]/[D1]", "XII/31"),
    ];
    for (picture, expected) in cases {
        assert_eq!(
            datetime::format_date_time(&context, &instant, picture, None, None, None)?,
            expected,
            "format-dateTime with '{picture}'"
        );
    }
    assert_eq!(datetime::format_integer(123, "0000")?, "0123");
    assert_eq!(datetime::format_integer(21, "1;o")?, "21st");
    assert_eq!(datetime::format_integer(57, "I")?, "LVII");
    Ok(())
}

#[test]
fn qname_functions() -> Result<(), AnyError> {
    let tree = Tree::parse_str("<a xmlns:p='urn:p' xmlns='urn:d'/>")?;
    let root = tree.document_element().ok_or("no document element")?;

    let resolved = qnames::resolve_qname("p:name", &root)?;
    assert_eq!(resolved.namespace(), Some("urn:p"));
    assert_eq!(
        qnames::resolve_qname("name", &root)?.namespace(),
        Some("urn:d")
    );
    assert!(
        qnames::resolve_qname("q:name", &root)
            .expect_err("unbound prefix")
            .is(FnErrorCode::Fons0004)
    );

    let built = qnames::qname(Some("http://www.example.com/example"), "ht:person")?;
    assert_eq!(qnames::prefix_from_qname(&built), Some("ht".to_owned()));
    assert_eq!(qnames::local_name_from_qname(&built), "person");
    assert_eq!(
        qnames::namespace_uri_from_qname(&built),
        "http://www.example.com/example"
    );
    assert_eq!(
        qnames::namespace_uri_for_prefix(Some("p"), &root),
        Some("urn:p".to_owned())
    );
    Ok(())
}

#[test]
fn node_functions() -> Result<(), AnyError> {
    let tree = Tree::parse_str("<a xml:lang='en'><b xml:id='x'/><c xml:lang='fr'/></a>")?;
    let root = tree.document_element().ok_or("no document element")?;
    let children = root.children();

    assert_eq!(nodes::name(&root), "a");
    assert_eq!(nodes::local_name(&root), "a");
    assert_eq!(nodes::namespace_uri(&root), "");
    assert_eq!(nodes::root(&children[0]), tree.root());
    assert!(nodes::has_children(&root));
    assert!(nodes::lang("en", &children[0]));
    assert!(!nodes::lang("en", &children[1]));
    assert_eq!(nodes::id(&["x".to_owned()], &root)?.len(), 1);
    assert_ne!(
        nodes::generate_id(&children[0]),
        nodes::generate_id(&children[1])
    );
    Ok(())
}

#[test]
fn sequence_functions_and_aggregates() -> Result<(), AnyError> {
    let context = FnContext::new();
    let items = integers(&[1, 2, 3, 4, 5]);

    assert!(!sequences::empty(&items));
    assert!(sequences::exists(&items));
    assert_eq!(rendered(&sequences::head(&items)), "1");
    assert_eq!(rendered(&sequences::tail(&items)), "2,3,4,5");
    assert_eq!(rendered(&sequences::reverse(&items)), "5,4,3,2,1");
    assert_eq!(rendered(&sequences::subsequence(&items, 4.0, None)), "4,5");
    assert_eq!(
        rendered(&sequences::subsequence(&items, 1.5, Some(2.6))),
        "2,3,4"
    );
    assert_eq!(rendered(&sequences::remove(&items, 1)), "2,3,4,5");
    assert_eq!(
        rendered(&sequences::insert_before(&items, 1, &integers(&[0]))),
        "0,1,2,3,4,5"
    );
    assert_eq!(sequences::count(&items), 5);

    assert_eq!(
        sequences::index_of(
            &context,
            &integers(&[10, 20, 30, 20]),
            &AtomicValue::integer(20_i64),
            None
        )?,
        vec![2, 4]
    );
    assert_eq!(
        rendered(&sequences::distinct_values(
            &context,
            &integers(&[1, 2, 1, 3]),
            None
        )?),
        "1,2,3"
    );

    assert_eq!(
        sequences::sum(&items, None)?.map(|value| value.string_value()),
        Some("15".to_owned())
    );
    assert_eq!(
        sequences::avg(&items)?.map(|value| value.string_value()),
        Some("3".to_owned())
    );
    assert_eq!(
        sequences::max(&context, &items, None)?.map(|value| value.string_value()),
        Some("5".to_owned())
    );
    assert_eq!(
        sequences::min(&context, &items, None)?.map(|value| value.string_value()),
        Some("1".to_owned())
    );

    assert!(
        sequences::exactly_one(&items)
            .expect_err("five items")
            .is(FnErrorCode::Forg0005)
    );
    assert!(
        sequences::one_or_more(&Sequence::empty())
            .expect_err("no items")
            .is(FnErrorCode::Forg0004)
    );
    Ok(())
}

#[test]
fn deep_equal_compares_atomic_values_and_trees() -> Result<(), AnyError> {
    let context = FnContext::new();
    let left = Tree::parse_str("<a><b>x</b></a>")?;
    let right = Tree::parse_str("<a><!-- ignored --><b>x</b></a>")?;
    assert!(sequences::deep_equal(
        &context,
        &Sequence::from(left.document_element().ok_or("no root")?),
        &Sequence::from(right.document_element().ok_or("no root")?),
        None,
    )?);
    assert!(!sequences::deep_equal(
        &context,
        &integers(&[1, 2]),
        &integers(&[2, 1]),
        None,
    )?);
    Ok(())
}

#[test]
fn collation_sensitive_comparisons_are_selectable() -> Result<(), AnyError> {
    let caseless = FnContext::new().with_default_collation(CASELESS_COLLATION)?;
    let codepoint = FnContext::new();

    assert_eq!(strings::compare(&caseless, "ÄÖÜ", "äöü", None)?, 0);
    assert_ne!(strings::compare(&codepoint, "ÄÖÜ", "äöü", None)?, 0);

    let mixed = Sequence::from_iter(["a", "A", "b"].map(AtomicValue::string));
    assert_eq!(
        sequences::distinct_values(&caseless, &mixed, None)?.len(),
        2
    );
    assert_eq!(
        sequences::distinct_values(&codepoint, &mixed, None)?.len(),
        3
    );
    Ok(())
}

#[test]
fn higher_order_functions() -> Result<(), AnyError> {
    let context = FnContext::new();
    let square = FunctionItem::anonymous(1, |_, arguments| {
        let value = arguments[0]
            .first()
            .and_then(Item::as_atomic)
            .cloned()
            .unwrap_or_else(|| AtomicValue::integer(0_i64));
        Ok(Sequence::from(numeric::multiply(&value, &value)?))
    });
    let plus = FunctionItem::anonymous(2, |_, arguments| {
        let left = arguments[0]
            .first()
            .and_then(Item::as_atomic)
            .cloned()
            .unwrap_or_else(|| AtomicValue::integer(0_i64));
        let right = arguments[1]
            .first()
            .and_then(Item::as_atomic)
            .cloned()
            .unwrap_or_else(|| AtomicValue::integer(0_i64));
        Ok(Sequence::from(numeric::add(&left, &right)?))
    });
    let is_even = FunctionItem::anonymous(1, |_, arguments| {
        let value = arguments[0]
            .first()
            .and_then(Item::as_atomic)
            .and_then(AtomicValue::as_f64)
            .unwrap_or_default();
        Ok(Sequence::from(AtomicValue::boolean(value % 2.0 == 0.0)))
    });

    let items = integers(&[1, 2, 3, 4, 5]);
    assert_eq!(
        rendered(&higher_order::for_each(&context, &items, &square)?),
        "1,4,9,16,25"
    );
    assert_eq!(
        rendered(&higher_order::filter(&context, &items, &is_even)?),
        "2,4"
    );
    assert_eq!(
        rendered(&higher_order::fold_left(
            &context,
            &items,
            &Sequence::from(AtomicValue::integer(0_i64)),
            &plus
        )?),
        "15"
    );
    assert_eq!(
        rendered(&higher_order::for_each_pair(
            &context,
            &integers(&[1, 2, 3]),
            &integers(&[10, 20, 30]),
            &plus
        )?),
        "11,22,33"
    );
    assert_eq!(
        rendered(&higher_order::sort(
            &context,
            &integers(&[3, 1, 2]),
            None,
            None
        )?),
        "1,2,3"
    );
    assert_eq!(
        rendered(&higher_order::apply(
            &context,
            &plus,
            &XdmArray::from_members([
                Sequence::from(AtomicValue::integer(2_i64)),
                Sequence::from(AtomicValue::integer(3_i64)),
            ])
        )?),
        "5"
    );
    assert_eq!(higher_order::function_arity(&plus), 2);
    assert!(higher_order::function_name(&plus).is_none());
    Ok(())
}

#[test]
fn map_functions() -> Result<(), AnyError> {
    let context = FnContext::new();
    let weekdays = XdmMap::new()
        .put(
            AtomicValue::integer(0_i64),
            Sequence::from(AtomicValue::string("Sonntag")),
        )
        .put(
            AtomicValue::integer(1_i64),
            Sequence::from(AtomicValue::string("Montag")),
        );

    assert_eq!(maps::size(&weekdays), 2);
    assert_eq!(maps::keys(&weekdays).len(), 2);
    assert!(maps::contains(&weekdays, &AtomicValue::integer(1_i64)));
    // op:same-key crosses the numeric tower.
    assert!(maps::contains(&weekdays, &AtomicValue::double(1.0)));
    assert_eq!(
        maps::get(&weekdays, &AtomicValue::integer(1_i64))
            .first()
            .map(ToString::to_string),
        Some("Montag".to_owned())
    );
    assert_eq!(
        maps::size(&maps::remove(&weekdays, &[AtomicValue::integer(0_i64)])),
        1
    );
    assert_eq!(
        maps::size(&maps::merge(
            &[
                weekdays.clone(),
                maps::entry(
                    AtomicValue::integer(2_i64),
                    Sequence::from(AtomicValue::string("Dienstag"))
                )
            ],
            DuplicateKeys::UseFirst
        )?),
        3
    );

    let names = FunctionItem::anonymous(2, |_, arguments| Ok(arguments[1].clone()));
    assert_eq!(maps::for_each(&context, &weekdays, &names)?.len(), 2);
    Ok(())
}

#[test]
fn array_functions() -> Result<(), AnyError> {
    let context = FnContext::new();
    let array = XdmArray::from_iter(["a", "b", "c", "d"].map(AtomicValue::string));

    assert_eq!(arrays::size(&array), 4);
    assert_eq!(
        arrays::get(&array, 2)?.first().map(ToString::to_string),
        Some("b".to_owned())
    );
    assert!(
        arrays::get(&array, 9)
            .expect_err("out of bounds")
            .is(FnErrorCode::Foay0001)
    );
    assert_eq!(arrays::size(&arrays::append(&array, Sequence::empty())), 5);
    assert_eq!(arrays::size(&arrays::subarray(&array, 2, Some(2))?), 2);
    assert_eq!(arrays::size(&arrays::remove(&array, &[1, 4])?), 2);
    assert_eq!(arrays::size(&arrays::reverse(&array)), 4);
    assert_eq!(
        arrays::head(&array)?.first().map(ToString::to_string),
        Some("a".to_owned())
    );

    let upper = FunctionItem::anonymous(1, |_, arguments| {
        let value = arguments[0]
            .first()
            .map(ToString::to_string)
            .unwrap_or_default();
        Ok(Sequence::from(AtomicValue::string(strings::upper_case(
            &value,
        ))))
    });
    let shouted = arrays::for_each(&context, &array, &upper)?;
    assert_eq!(
        arrays::get(&shouted, 1)?.first().map(ToString::to_string),
        Some("A".to_owned())
    );
    Ok(())
}

#[test]
fn parse_and_serialize_round_trip() -> Result<(), AnyError> {
    let document = xmlio::parse_xml(
        "<catalog xmlns:dc='urn:dc'><book dc:id='b1'>Dune &amp; Emma</book></catalog>",
        Some("http://example.com/catalog.xml"),
    )?;
    // F&O 3.1 §14.7.1: the base URI is the caller's, the document URI is absent.
    assert_eq!(
        document.base_uri().as_deref(),
        Some("http://example.com/catalog.xml")
    );
    assert_eq!(document.document_uri(), None);

    let serialized = xmlio::serialize(
        &Sequence::from(document.children().into_iter().next().ok_or("no root")?),
        &xmlio::SerializationParameters::default(),
    )?;
    assert_eq!(
        serialized,
        r#"<catalog xmlns:dc="urn:dc"><book dc:id="b1">Dune &amp; Emma</book></catalog>"#
    );

    // A fragment may hold several top-level nodes.
    let fragment = xmlio::parse_xml_fragment("<a/>text<b/>", None)?;
    assert_eq!(fragment.children().len(), 3);
    assert_eq!(
        xmlio::serialize(
            &Sequence::from(fragment),
            &xmlio::SerializationParameters::default()
        )?,
        "<a/>text<b/>"
    );
    Ok(())
}

#[test]
fn errors_and_trace() -> Result<(), AnyError> {
    let sink = Arc::new(CollectingTrace::new());
    let context = FnContext::new().with_trace_sink(sink.clone());

    let value = Sequence::from(AtomicValue::integer(7_i64));
    assert_eq!(diagnostics::trace(&context, &value, "seven: "), value);
    assert_eq!(sink.messages(), vec!["seven: 7".to_owned()]);

    let raised = diagnostics::error(None, Some("boom"), None);
    assert!(raised.is(FnErrorCode::Foer0000));
    assert_eq!(raised.description(), "boom");
    Ok(())
}

#[test]
fn the_registry_binds_names_to_callables() -> Result<(), AnyError> {
    let catalogue = registry::builtins();
    let context = FnContext::new();

    let name = ExpandedName::new(Some(registry::FN_NAMESPACE), "string-join");
    let string_join = catalogue.lookup(&name, 2).ok_or("fn:string-join#2")?;
    let joined = string_join.call(
        &context,
        &[
            Sequence::from_iter(["a", "b", "c"].map(AtomicValue::string)),
            Sequence::from(AtomicValue::string("-")),
        ],
    )?;
    assert_eq!(rendered(&joined), "a-b-c");

    // Arity is part of the key.
    assert!(catalogue.lookup(&name, 7).is_none());
    Ok(())
}
