use super::*;

/// Collects an XML-mode attribute iteration into owned key/value pairs.
#[allow(clippy::type_complexity)]
fn collect_xml(buf: &str, pos: usize) -> Vec<Result<(Vec<u8>, Vec<u8>), AttrError>> {
    Attributes::new(buf, pos)
        .map(|r| r.map(|a| (a.key.as_ref().to_vec(), a.value.to_vec())))
        .collect()
}

#[test]
fn single_and_double_quoted() {
    assert_eq!(
        collect_xml("e a='1' b=\"2\"", 1),
        vec![
            Ok((b"a".to_vec(), b"1".to_vec())),
            Ok((b"b".to_vec(), b"2".to_vec())),
        ]
    );
}

#[test]
fn whitespace_around_eq() {
    assert_eq!(
        collect_xml("e   a   =   'v'   b='w'", 1),
        vec![
            Ok((b"a".to_vec(), b"v".to_vec())),
            Ok((b"b".to_vec(), b"w".to_vec())),
        ]
    );
}

#[test]
fn empty_value() {
    assert_eq!(
        collect_xml("e a=''", 1),
        vec![Ok((b"a".to_vec(), b"".to_vec()))]
    );
}

#[test]
fn value_may_contain_other_quote() {
    assert_eq!(
        collect_xml("e a=\"it's\" b='say \"hi\"'", 1),
        vec![
            Ok((b"a".to_vec(), b"it's".to_vec())),
            Ok((b"b".to_vec(), b"say \"hi\"".to_vec())),
        ]
    );
}

#[test]
fn key_only_is_error_in_xml() {
    let mut iter = Attributes::new("e novalue", 1);
    assert_eq!(iter.next(), Some(Err(AttrError::ExpectedEq(9))));
    assert_eq!(iter.next(), None);
}

#[test]
fn unquoted_value_is_error_in_xml() {
    let mut iter = Attributes::new("e a=b c='d'", 1);
    assert_eq!(iter.next(), Some(Err(AttrError::UnquotedValue(4))));
    // Recovery skips the bad value and continues.
    assert_eq!(
        iter.next(),
        Some(Ok(Attribute::from(("c".as_bytes(), "d".as_bytes()))))
    );
}

#[test]
fn missing_closing_quote() {
    let mut iter = Attributes::new("e a='unterminated", 1);
    assert_eq!(iter.next(), Some(Err(AttrError::ExpectedQuote(17, b'\''))));
    assert_eq!(iter.next(), None);
}

#[test]
fn missing_value_after_eq() {
    let mut iter = Attributes::new("e a=", 1);
    assert_eq!(iter.next(), Some(Err(AttrError::ExpectedValue(4))));
}

#[test]
fn duplicate_detection_on_by_default() {
    let mut iter = Attributes::new("e a='1' a='2'", 1);
    assert_eq!(
        iter.next(),
        Some(Ok(Attribute::from(("a".as_bytes(), "1".as_bytes()))))
    );
    // Position of second `a` key is 8, previous at 2.
    assert_eq!(iter.next(), Some(Err(AttrError::Duplicated(8, 2))));
}

#[test]
fn duplicate_detection_can_be_disabled() {
    let mut iter = Attributes::new("e a='1' a='2'", 1);
    iter.with_checks(false);
    assert_eq!(
        iter.next(),
        Some(Ok(Attribute::from(("a".as_bytes(), "1".as_bytes()))))
    );
    assert_eq!(
        iter.next(),
        Some(Ok(Attribute::from(("a".as_bytes(), "2".as_bytes()))))
    );
}

#[test]
fn duplicate_detection_past_hash_threshold() {
    // Build a tag with more than SMALL_ATTRIBUTE_COUNT unique attributes, then a
    // duplicate; the hashed path must still catch it.
    let mut tag = String::from("e");
    for i in 0..40 {
        tag.push_str(&format!(" a{}='v'", i));
    }
    tag.push_str(" a0='dup'");
    let last = Attributes::new(&tag, 1).last().expect("at least one item");
    assert!(matches!(last, Err(AttrError::Duplicated(_, _))));
}

#[test]
fn html_mode_unquoted_and_empty() {
    let mut iter = Attributes::html("e a=b flag c = 'd'", 1);
    assert_eq!(
        iter.next(),
        Some(Ok(Attribute::from(("a".as_bytes(), "b".as_bytes()))))
    );
    assert_eq!(
        iter.next(),
        Some(Ok(Attribute {
            key: QName(b"flag"),
            value: Cow::Borrowed(b""),
        }))
    );
    assert_eq!(
        iter.next(),
        Some(Ok(Attribute::from(("c".as_bytes(), "d".as_bytes()))))
    );
    assert_eq!(iter.next(), None);
}

#[test]
fn attr_span_accessors() {
    let attr: Attr<&[u8]> = Attr::DoubleQ(b"k".as_ref(), b"v".as_ref());
    assert_eq!(attr.key(), QName(b"k"));
    assert_eq!(attr.value(), b"v");
    let empty: Attr<&[u8]> = Attr::Empty(b"k".as_ref());
    assert_eq!(empty.value(), b"");
    // `Attr<u8>` has no `Debug` (bound requires `AsRef<[u8]>`), so compare with `==`.
    let mapped = Attr::SingleQ(1u8, 2u8).map(|x| x + 10);
    assert!(mapped == Attr::SingleQ(11, 12));
    let tuple: (u8, Option<u8>) = Attr::Empty(7u8).into();
    assert_eq!(tuple, (7, None));
}

#[test]
fn normalized_value_unescapes_and_normalizes() -> XmlResult<()> {
    let attr = Attribute {
        key: QName(b"k"),
        value: Cow::Borrowed(b"a &amp; b\tc"),
    };
    assert_eq!(attr.normalized_value(XmlVersion::Implicit1_0)?, "a & b c");
    Ok(())
}

#[test]
fn decoded_and_normalized_value() -> XmlResult<()> {
    let attr = Attribute {
        key: QName(b"k"),
        value: Cow::Borrowed(b"x&lt;y"),
    };
    assert_eq!(
        attr.decoded_and_normalized_value(XmlVersion::Implicit1_0, Decoder::utf8())?,
        "x<y"
    );
    Ok(())
}

#[test]
fn as_bool_values() {
    assert_eq!(Attribute::from(("k", "true")).as_bool(), Some(true));
    assert_eq!(Attribute::from(("k", "0")).as_bool(), Some(false));
    assert_eq!(Attribute::from(("k", "maybe")).as_bool(), None);
}

#[test]
fn from_bytes_does_not_escape() {
    let attr = Attribute::from(("k".as_bytes(), "a & b".as_bytes()));
    assert_eq!(attr.value.as_ref(), b"a & b");
    // From (&str, &str) escapes the value.
    let attr = Attribute::from(("k", "a & b"));
    assert_eq!(attr.value.as_ref(), b"a &amp; b");
}
