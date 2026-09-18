//! `fn:deep-equal` across every item kind.

use oxixml_infoset::{
    AtomicType, AtomicValue, ExpandedName, HTML_ASCII_CASE_INSENSITIVE_COLLATION, Item, Sequence,
    Tree, TreeBuilder, XdmArray, XdmError, XdmMap, XdmNode, deep_equal, deep_equal_items,
    deep_equal_with_collation,
};

fn root_of(xml: &str) -> Result<Sequence, XdmError> {
    let tree = Tree::parse_str(xml)?;
    Ok(Sequence::from(tree.document_element().ok_or_else(
        || XdmError::new(oxixml_infoset::ErrorCode::Fodc0006, "no root"),
    )?))
}

fn equal(left: &str, right: &str) -> Result<bool, XdmError> {
    deep_equal(&root_of(left)?, &root_of(right)?)
}

#[test]
fn identical_documents_are_deep_equal() -> Result<(), XdmError> {
    assert!(equal("<a x='1'><b>t</b></a>", "<a x='1'><b>t</b></a>")?);
    Ok(())
}

#[test]
fn comments_and_processing_instructions_are_ignored_inside_content() -> Result<(), XdmError> {
    assert!(equal(
        "<a><b>t</b></a>",
        "<a><!--note--><b>t</b><?pi data?></a>"
    )?);
    assert!(
        equal("<a><!--x--></a>", "<a><!--y--></a>")?,
        "two comments that are only children of an element are both ignored"
    );
    Ok(())
}

#[test]
fn comments_compared_directly_are_not_ignored() -> Result<(), XdmError> {
    let one = Tree::parse_str("<a><!--x--></a>")?;
    let two = Tree::parse_str("<a><!--y--></a>")?;
    let left = one
        .document_element()
        .and_then(|a| a.children().first().cloned())
        .expect("a comment");
    let right = two
        .document_element()
        .and_then(|a| a.children().first().cloned())
        .expect("a comment");
    assert!(!deep_equal_items(
        &Item::from(left),
        &Item::from(right),
        None
    )?);
    Ok(())
}

#[test]
fn attribute_order_is_irrelevant_but_values_are_not() -> Result<(), XdmError> {
    assert!(equal(r#"<a p="1" q="2"/>"#, r#"<a q="2" p="1"/>"#)?);
    assert!(!equal(r#"<a p="1" q="2"/>"#, r#"<a p="1" q="9"/>"#)?);
    assert!(!equal(r#"<a p="1"/>"#, r#"<a p="1" q="2"/>"#)?);
    Ok(())
}

#[test]
fn names_namespaces_and_prefixes() -> Result<(), XdmError> {
    // The prefix is not part of a name; the namespace is.
    assert!(equal(
        r#"<p:a xmlns:p="urn:x"/>"#,
        r#"<q:a xmlns:q="urn:x"/>"#
    )?);
    assert!(!equal(
        r#"<p:a xmlns:p="urn:x"/>"#,
        r#"<p:a xmlns:p="urn:y"/>"#
    )?);
    assert!(!equal("<a/>", "<b/>")?);

    // Namespace nodes themselves are not compared, so an unused declaration does not matter.
    assert!(equal(r#"<a xmlns:unused="urn:z"/>"#, "<a/>")?);
    Ok(())
}

#[test]
fn text_content_and_child_order_matter() -> Result<(), XdmError> {
    assert!(!equal("<a>x</a>", "<a>y</a>")?);
    assert!(!equal("<a><b/><c/></a>", "<a><c/><b/></a>")?);
    assert!(
        equal("<a>x<b/>y</a>", "<a>x<b/>y</a>")?,
        "mixed content compares child by child"
    );
    assert!(
        !equal("<a>xy</a>", "<a>x<b/>y</a>")?,
        "a different number of children"
    );
    Ok(())
}

#[test]
fn documents_and_kinds() -> Result<(), XdmError> {
    let one = Tree::parse_str("<!--c--><a/>")?;
    let two = Tree::parse_str("<a/><?pi?>")?;
    assert!(
        deep_equal(&Sequence::from(one.root()), &Sequence::from(two.root()))?,
        "a document node ignores its comment and processing-instruction children too"
    );

    // Different kinds are never deep-equal.
    let element = one.document_element().expect("a document element");
    assert!(!deep_equal_items(
        &Item::from(one.root()),
        &Item::from(element.clone()),
        None
    )?);
    assert!(!deep_equal_items(
        &Item::from(element),
        &Item::from(AtomicValue::string("a")),
        None
    )?);
    Ok(())
}

#[test]
fn sequences_must_have_the_same_length_and_pairwise_equal_items() -> Result<(), XdmError> {
    let a = Sequence::from_items([
        Item::from(AtomicValue::integer(1_i64)),
        Item::from(AtomicValue::string("x")),
    ]);
    let b = Sequence::from_items([
        Item::from(AtomicValue::decimal(1_i64)),
        Item::from(AtomicValue::any_uri("x")),
    ]);
    assert!(
        deep_equal(&a, &b)?,
        "items are compared with `eq`, which crosses type boundaries within a family"
    );

    assert!(!deep_equal(&a, &Sequence::empty())?);
    assert!(deep_equal(&Sequence::empty(), &Sequence::empty())?);
    Ok(())
}

#[test]
fn atomic_edge_cases() -> Result<(), XdmError> {
    let nan = Sequence::from(AtomicValue::double(f64::NAN));
    assert!(
        deep_equal(&nan, &nan)?,
        "NaN is deep-equal to NaN even though it is never `eq` to it"
    );

    // Values whose types are not comparable are simply not deep-equal — no type error.
    let integer = Sequence::from(AtomicValue::integer(1_i64));
    let string = Sequence::from(AtomicValue::string("1"));
    assert!(!deep_equal(&integer, &string)?);

    // untypedAtomic compares as a string.
    assert!(deep_equal(
        &Sequence::from(AtomicValue::untyped_atomic("1")),
        &string
    )?);

    let date = Sequence::from(AtomicValue::parse(AtomicType::Date, "2020-01-01")?);
    assert!(!deep_equal(&date, &integer)?);
    assert!(deep_equal(&date, &date)?);
    Ok(())
}

#[test]
fn collations_are_honoured_for_text_and_attributes() -> Result<(), XdmError> {
    let upper = root_of("<a>TEXT</a>")?;
    let lower = root_of("<a>text</a>")?;
    assert!(!deep_equal(&upper, &lower)?);
    assert!(deep_equal_with_collation(
        &upper,
        &lower,
        Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION)
    )?);

    let attribute_upper = root_of(r#"<a k="VALUE"/>"#)?;
    let attribute_lower = root_of(r#"<a k="value"/>"#)?;
    assert!(!deep_equal(&attribute_upper, &attribute_lower)?);
    assert!(deep_equal_with_collation(
        &attribute_upper,
        &attribute_lower,
        Some(HTML_ASCII_CASE_INSENSITIVE_COLLATION)
    )?);
    Ok(())
}

#[test]
fn arrays_and_maps() -> Result<(), XdmError> {
    let a = XdmArray::from_iter([AtomicValue::integer(1_i64), AtomicValue::integer(2_i64)]);
    let b = XdmArray::from_iter([AtomicValue::decimal(1_i64), AtomicValue::double(2.0)]);
    let c = XdmArray::from_iter([AtomicValue::integer(2_i64), AtomicValue::integer(1_i64)]);

    assert!(deep_equal_items(
        &Item::from(a.clone()),
        &Item::from(b),
        None
    )?);
    assert!(!deep_equal_items(
        &Item::from(a.clone()),
        &Item::from(c),
        None
    )?);
    assert!(!deep_equal_items(
        &Item::from(a.clone()),
        &Item::from(XdmArray::new()),
        None
    )?);

    let first = XdmMap::from_iter([
        (
            AtomicValue::string("k"),
            Sequence::from(AtomicValue::integer(1_i64)),
        ),
        (
            AtomicValue::integer(2_i64),
            Sequence::from(AtomicValue::string("two")),
        ),
    ]);
    let reordered = XdmMap::from_iter([
        (
            AtomicValue::double(2.0),
            Sequence::from(AtomicValue::string("two")),
        ),
        (
            AtomicValue::any_uri("k"),
            Sequence::from(AtomicValue::decimal(1_i64)),
        ),
    ]);
    assert!(
        deep_equal_items(&Item::from(first.clone()), &Item::from(reordered), None)?,
        "map keys match under op:same-key and order is irrelevant"
    );

    let different = XdmMap::from_iter([(
        AtomicValue::string("k"),
        Sequence::from(AtomicValue::integer(9_i64)),
    )]);
    assert!(!deep_equal_items(
        &Item::from(first),
        &Item::from(different),
        None
    )?);

    // A map is never deep-equal to an array.
    assert!(!deep_equal_items(
        &Item::from(XdmMap::new()),
        &Item::from(a),
        None
    )?);
    Ok(())
}

#[test]
fn nilled_elements_differ_from_ordinary_empty_ones() -> Result<(), XdmError> {
    use oxixml_infoset::Annotation;

    let build = |nilled: Option<bool>| -> Result<Sequence, XdmError> {
        let mut builder = TreeBuilder::new();
        builder.start_annotated_element(
            ExpandedName::new(None::<&str>, "e"),
            Annotation {
                nilled,
                ..Annotation::default()
            },
        )?;
        builder.end_element()?;
        let tree = builder.build()?;
        Ok(Sequence::from(
            tree.document_element().expect("a document element"),
        ))
    };

    assert!(deep_equal(&build(Some(true))?, &build(Some(true))?)?);
    assert!(!deep_equal(&build(Some(true))?, &build(Some(false))?)?);
    assert!(
        deep_equal(&build(None)?, &build(Some(false))?)?,
        "an unannotated element reports `nilled` as false"
    );
    Ok(())
}

#[test]
fn deep_equal_reaches_arbitrarily_deep() -> Result<(), XdmError> {
    let mut left = String::new();
    let mut right = String::new();
    for depth in 0..50 {
        left.push_str(&format!("<n{depth}>"));
        right.push_str(&format!("<n{depth}>"));
    }
    left.push_str("leaf");
    right.push_str("leaf");
    for depth in (0..50).rev() {
        left.push_str(&format!("</n{depth}>"));
        right.push_str(&format!("</n{depth}>"));
    }
    assert!(equal(&left, &right)?);

    let mut differing = right.replace("leaf", "other");
    differing.truncate(differing.len());
    assert!(!equal(&left, &differing)?);
    Ok(())
}
