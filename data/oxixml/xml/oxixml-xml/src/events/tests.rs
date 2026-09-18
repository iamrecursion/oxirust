use super::*;
use crate::name::QName;

#[test]
fn start_name_and_attributes() {
    let start = BytesStart::from_content("tag a='1' b=\"2\"", 3);
    assert_eq!(start.name(), QName(b"tag"));
    assert_eq!(start.attributes_raw(), b" a='1' b=\"2\"");
    let attrs: Vec<_> = start
        .attributes()
        .map(|a| a.expect("attr"))
        .map(|a| (a.key.as_ref().to_vec(), a.value.to_vec()))
        .collect();
    assert_eq!(
        attrs,
        vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"b".to_vec(), b"2".to_vec())
        ]
    );
}

#[test]
fn start_push_and_clear_attributes() {
    let mut start = BytesStart::new("e");
    start.push_attribute(("k", "v & w"));
    // Value is escaped when built from &str.
    assert_eq!(start.attributes_raw(), b" k=\"v &amp; w\"");
    start.clear_attributes();
    assert_eq!(start.attributes_raw(), b"");
    assert_eq!(start.name(), QName(b"e"));
}

#[test]
fn start_set_name_and_to_end() {
    let mut start = BytesStart::new("old");
    start.set_name(b"new");
    assert_eq!(start.name(), QName(b"new"));
    assert_eq!(start.to_end().name(), QName(b"new"));
}

#[test]
fn start_try_get_attribute() -> Result<(), AttrError> {
    let start = BytesStart::from_content("e a='1' b='2'", 1);
    assert_eq!(
        start.try_get_attribute(b"b")?.map(|a| a.value.to_vec()),
        Some(b"2".to_vec())
    );
    assert!(start.try_get_attribute(b"missing")?.is_none());
    Ok(())
}

#[test]
fn start_owns_and_borrows() {
    let start = BytesStart::new("e");
    let owned: BytesStart<'static> = start.to_owned();
    assert_eq!(owned.name(), QName(b"e"));
    let borrowed = owned.borrow();
    assert_eq!(borrowed.name(), QName(b"e"));
    assert_eq!(start.into_owned().name(), QName(b"e"));
}

#[test]
fn end_accessors() {
    let end = BytesEnd::new("ns:tag");
    assert_eq!(end.name(), QName(b"ns:tag"));
    assert_eq!(end.local_name().as_ref(), b"tag");
    assert_eq!(&end as &[u8], b"ns:tag");
}

#[test]
fn text_escaping_and_decode() -> Result<(), EncodingError> {
    // `new` escapes; the stored bytes are escaped.
    let text = BytesText::new("a < b & c");
    assert_eq!(&text as &[u8], b"a &lt; b &amp; c");
    // `decode` returns the raw (still escaped) content.
    assert_eq!(text.decode()?, "a &lt; b &amp; c");
    // `from_escaped` stores verbatim.
    let raw = BytesText::from_escaped("a &lt; b");
    assert_eq!(&raw as &[u8], b"a &lt; b");
    Ok(())
}

#[test]
fn text_eol_normalization() -> Result<(), EncodingError> {
    let text = BytesText::from_escaped("line1\r\nline2\rline3");
    assert_eq!(text.xml10_content()?, "line1\nline2\nline3");
    assert_eq!(
        text.xml_content(XmlVersion::Explicit1_0)?,
        "line1\nline2\nline3"
    );
    Ok(())
}

#[test]
fn text_inplace_trim() {
    let mut text = BytesText::from_escaped("   ");
    assert!(text.inplace_trim_start());
    let mut text = BytesText::from_escaped("  hi  ");
    assert!(!text.inplace_trim_start());
    assert!(!text.inplace_trim_end());
    assert_eq!(&text as &[u8], b"hi");
}

#[test]
fn cdata_escaped_split_and_escape() -> Result<(), EncodingError> {
    let pieces = BytesCData::escaped("a]]>b").collect::<Vec<_>>();
    assert_eq!(pieces, vec![BytesCData::new("a]]"), BytesCData::new(">b")]);

    let cdata = BytesCData::new("<tag> & 'x'");
    let text = cdata.escape()?;
    assert_eq!(&text as &[u8], b"&lt;tag&gt; &amp; &apos;x&apos;");
    Ok(())
}

#[test]
fn pi_target_content_attributes() {
    let pi = BytesPI::new(r#"xml-stylesheet href="style.css" type="text/css""#);
    assert_eq!(pi.target(), b"xml-stylesheet");
    assert_eq!(pi.content(), br#" href="style.css" type="text/css""#);
    let attrs: Vec<_> = pi
        .attributes()
        .map(|a| a.expect("attr"))
        .map(|a| (a.key.as_ref().to_vec(), a.value.to_vec()))
        .collect();
    assert_eq!(
        attrs,
        vec![
            (b"href".to_vec(), b"style.css".to_vec()),
            (b"type".to_vec(), b"text/css".to_vec()),
        ]
    );
}

#[test]
fn decl_new_and_accessors() -> Result<(), Error> {
    let decl = BytesDecl::new("1.0", Some("UTF-8"), Some("yes"));
    assert_eq!(decl.version()?.as_ref(), b"1.0");
    assert_eq!(
        decl.encoding().transpose()?.map(|c| c.to_vec()),
        Some(b"UTF-8".to_vec())
    );
    assert_eq!(
        decl.standalone().transpose()?.map(|c| c.to_vec()),
        Some(b"yes".to_vec())
    );
    assert_eq!(decl.xml_version()?, XmlVersion::Explicit1_0);
    Ok(())
}

#[test]
fn decl_missing_version() {
    let decl = BytesDecl::from_start(BytesStart::from_content(" encoding='utf-8'", 0));
    match decl.version() {
        Err(Error::IllFormed(IllFormedError::MissingDeclVersion(Some(key)))) => {
            assert_eq!(key, "encoding")
        }
        other => panic!("unexpected: {other:?}"),
    }
    let decl = BytesDecl::from_start(BytesStart::from_content(" version='1.2'", 0));
    assert!(matches!(
        decl.xml_version(),
        Err(Error::IllFormed(IllFormedError::UnknownVersion))
    ));
}

#[test]
fn ref_char_and_general() -> Result<(), Error> {
    assert!(BytesRef::new("#x41").is_char_ref());
    assert!(!BytesRef::new("amp").is_char_ref());
    assert_eq!(BytesRef::new("#x41").resolve_char_ref()?, Some('A'));
    assert_eq!(BytesRef::new("#65").resolve_char_ref()?, Some('A'));
    assert_eq!(BytesRef::new("amp").resolve_char_ref()?, None);
    Ok(())
}

#[test]
fn event_deref_and_into_owned() {
    let event = Event::Start(BytesStart::from_content("e a='1'", 1));
    assert_eq!(&*event, b"e a='1'");
    let owned: Event<'static> = event.into_owned();
    assert!(matches!(owned, Event::Start(_)));
    assert_eq!(&*Event::Eof, b"");
}

#[test]
fn event_borrow_roundtrip() {
    let text = Event::Text(BytesText::from_escaped("hi"));
    let borrowed = text.borrow();
    assert_eq!(&*borrowed, b"hi");
}
