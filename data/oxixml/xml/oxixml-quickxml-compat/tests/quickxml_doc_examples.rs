//! Compiles and runs realistic usage snippets lifted from the `quick-xml`
//! 0.41 rustdoc (`src/name.rs`, `src/writer.rs`, `src/reader/ns_reader.rs`,
//! `src/events/mod.rs`, `src/events/attributes.rs`) against
//! `oxixml-quickxml-compat`, proving path-for-path compatibility beyond the
//! `README.md` reader/writer loop already covered in
//! `quickxml_readme_examples.rs`.
//!
//! Each test names the upstream doc comment it reproduces so the mapping back
//! to `quick-xml` 0.41 is traceable.

use oxixml_quickxml_compat as quick_xml;

/// `quick_xml::name::QName::local_name` doc example.
#[test]
fn qname_local_name() {
    use quick_xml::name::QName;

    let simple = QName(b"simple-name");
    assert_eq!(simple.local_name().as_ref(), b"simple-name");

    let qname = QName(b"namespace:simple-name");
    assert_eq!(qname.local_name().as_ref(), b"simple-name");
}

/// `quick_xml::name::QName::prefix` doc example.
#[test]
fn qname_prefix() {
    use quick_xml::name::QName;

    let simple = QName(b"simple-name");
    assert_eq!(simple.prefix(), None);

    let qname = QName(b"prefix:simple-name");
    assert_eq!(
        qname.prefix().as_ref().map(|n| n.as_ref()),
        Some(b"prefix".as_ref())
    );
}

/// `quick_xml::name::QName::as_namespace_binding` doc example.
#[test]
fn qname_as_namespace_binding() {
    use quick_xml::name::{PrefixDeclaration, QName};

    let qname = QName(b"xmlns");
    assert_eq!(
        qname.as_namespace_binding(),
        Some(PrefixDeclaration::Default)
    );

    let qname = QName(b"xmlns:prefix");
    assert_eq!(
        qname.as_namespace_binding(),
        Some(PrefixDeclaration::Named(b"prefix"))
    );

    let qname = QName(b"other-name");
    assert_eq!(qname.as_namespace_binding(), None);
}

/// `quick_xml::writer::Config::add_space_before_slash_in_empty_elements` doc
/// example: uses [`quick_xml::writer::Writer::create_element`] to build an
/// empty element with an attribute.
#[test]
fn writer_config_add_space_before_slash() {
    use quick_xml::reader::Reader;
    use quick_xml::writer::Writer;
    use std::io::Cursor;

    let mut writer = Writer::new(Cursor::new(Vec::new()));
    writer.config_mut().add_space_before_slash_in_empty_elements = true;

    writer
        .create_element("tag")
        .with_attribute(("attr1", "value1"))
        .write_empty()
        .expect("write empty");

    let result = writer.into_inner().into_inner();
    let expected = r#"<tag attr1="value1" />"#;
    assert_eq!(result, expected.as_bytes());

    // Reader::from_reader is unused here except to prove the `reader` module
    // path also resolves under the same import block.
    let _reader = Reader::from_str("<a/>");
}

/// `quick_xml::reader::NsReader::read_resolved_event` doc example: reads a
/// namespaced document and resolves prefixes to their bound namespace URIs.
#[test]
fn ns_reader_read_resolved_event() {
    use quick_xml::events::Event;
    use quick_xml::name::{Namespace, QName, ResolveResult::*};
    use quick_xml::reader::NsReader;

    let mut reader = NsReader::from_str(
        r#"
        <x:tag1 xmlns:x="www.xxxx" xmlns:y="www.yyyy" att1 = "test">
           <y:tag2><!--Test comment-->Test</y:tag2>
           <y:tag2>Test 2</y:tag2>
        </x:tag1>
    "#,
    );
    reader.config_mut().trim_text(true);

    let mut count = 0;
    let mut txt = Vec::new();
    loop {
        match reader.read_resolved_event().expect("event") {
            (Bound(Namespace(b"www.xxxx")), Event::Start(e)) => {
                count += 1;
                assert_eq!(e.local_name(), QName(b"tag1").into());
            }
            (Bound(Namespace(b"www.yyyy")), Event::Start(e)) => {
                count += 1;
                assert_eq!(e.local_name(), QName(b"tag2").into());
            }
            (_, Event::Start(_)) => unreachable!(),

            (_, Event::Text(e)) => txt.push(e.decode().expect("utf-8").into_owned()),
            (_, Event::Eof) => break,
            _ => (),
        }
    }
    assert_eq!(count, 3);
    assert_eq!(txt, vec!["Test".to_string(), "Test 2".to_string()]);
}

/// `quick_xml::events::BytesStart` doc example: building a start tag,
/// inspecting its name and iterating over its attributes.
#[test]
fn bytes_start_and_attributes() {
    use quick_xml::events::{BytesStart, Event};
    use quick_xml::name::QName;

    let start = BytesStart::new("greeting").with_attributes([("lang", "en")]);
    assert_eq!(start.name(), QName(b"greeting"));

    let (key, value) = {
        let attr = start
            .attributes()
            .next()
            .expect("one attribute")
            .expect("valid attribute");
        (attr.key.as_ref().to_vec(), attr.value.to_vec())
    };
    assert_eq!(
        (key.as_slice(), value.as_slice()),
        (&b"lang"[..], &b"en"[..])
    );

    let event = Event::Start(start);
    assert_eq!(&*event, b"greeting lang=\"en\"".as_ref());
}

/// `quick_xml::escape::escape` / `unescape` doc example.
#[test]
fn escape_and_unescape_round_trip() {
    use quick_xml::escape::{escape, unescape};

    assert_eq!(escape("a < b & c"), "a &lt; b &amp; c");
    assert_eq!(unescape("a &lt; b &amp; c").expect("valid"), "a < b & c");
}

/// `quick_xml::events::attributes::Attributes` iteration, as used throughout
/// the `quick-xml` 0.41 reader examples.
#[test]
fn attributes_iteration() {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_str(r#"<a k1="v1" k2="v2"/>"#);
    let event = reader.read_event().expect("event");
    let Event::Empty(start) = event else {
        panic!("expected an empty element event");
    };

    let mut pairs = Vec::new();
    for attr in start.attributes() {
        let attr = attr.expect("valid attribute");
        pairs.push((
            String::from_utf8(attr.key.as_ref().to_vec()).expect("utf-8"),
            String::from_utf8(attr.value.to_vec()).expect("utf-8"),
        ));
    }
    assert_eq!(
        pairs,
        vec![
            ("k1".to_string(), "v1".to_string()),
            ("k2".to_string(), "v2".to_string()),
        ]
    );
}

/// `quick_xml::writer::Writer::create_element` / `ElementWriter` doc-style
/// example: nested elements built with the fluent `ElementWriter` API.
#[test]
fn element_writer_nested_content() {
    use quick_xml::events::BytesText;
    use quick_xml::writer::Writer;

    let mut writer = Writer::new(Vec::new());
    writer
        .create_element("outer")
        .write_inner_content(|w| {
            w.create_element("inner")
                .write_text_content(BytesText::new("hello"))?;
            Ok(())
        })
        .expect("write inner content");

    let result = writer.into_inner();
    assert_eq!(result, b"<outer><inner>hello</inner></outer>");
}

/// `Reader::read_to_end` skip-content pattern, from the `quick-xml` 0.41
/// `read_to_end_into` doc example.
#[test]
fn reader_read_to_end_skips_nested_content() {
    use quick_xml::events::{BytesStart, Event};
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_str(
        r#"
        <outer>
            <inner>
                <inner></inner>
                <inner/>
                <outer></outer>
                <outer/>
            </inner>
        </outer>
    "#,
    );
    reader.config_mut().trim_text(true);

    let start = BytesStart::new("outer");
    let end = start.to_end().into_owned();

    assert_eq!(reader.read_event().expect("start"), Event::Start(start));

    reader.read_to_end(end.name()).expect("skip to end");

    assert_eq!(reader.read_event().expect("eof"), Event::Eof);
}
