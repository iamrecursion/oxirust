//! Tests for the events-based [`Writer`](crate::writer::Writer).

use crate::events::{BytesCData, BytesEnd, BytesPI, BytesStart, BytesText, Event};
use crate::reader::Reader;
use crate::writer::Writer;

/// Collects every event of a document into owned events (excluding the final
/// `Eof`).
fn read_all(xml: &str) -> Vec<Event<'static>> {
    let mut reader = Reader::from_str(xml);
    let mut events = Vec::new();
    loop {
        match reader.read_event().expect("valid event") {
            Event::Eof => break,
            e => events.push(e.into_owned()),
        }
    }
    events
}

/// Writes a stream of events with a non-indenting writer and returns the bytes.
fn write_all(events: &[Event<'_>]) -> Vec<u8> {
    let mut writer = Writer::new(Vec::new());
    for event in events {
        writer.write_event(event.borrow()).expect("write");
    }
    writer.into_inner()
}

/// Reads a document, writes it back verbatim, then re-reads it and asserts the
/// event stream is preserved and (for already-minified input) the bytes are
/// identical.
fn assert_round_trip(xml: &str) {
    let events = read_all(xml);
    let written = write_all(&events);
    assert_eq!(
        std::str::from_utf8(&written).expect("utf-8"),
        xml,
        "written bytes differ from the input",
    );
    let reparsed = read_all(std::str::from_utf8(&written).expect("utf-8"));
    assert_eq!(events, reparsed, "event stream changed after a round trip");
}

#[test]
fn round_trip_corpus() {
    assert_round_trip(r#"<root>text</root>"#);
    assert_round_trip(r#"<a k1="v1" k2="v2"><b/><c>x</c></a>"#);
    assert_round_trip(r#"<x:a xmlns:x="urn:x"><x:b>t</x:b></x:a>"#);
    assert_round_trip(r#"<a><!--a comment--></a>"#);
    assert_round_trip(r#"<a><![CDATA[ raw <b> & ]]></a>"#);
    assert_round_trip(r#"<a>one &amp; two &lt; three</a>"#);
    assert_round_trip(r#"<?xml version="1.0" encoding="UTF-8"?><root/>"#);
    assert_round_trip(r#"<a><?target instruction?></a>"#);
    assert_round_trip("<a>  \n  <b/>\n</a>");
    assert_round_trip(r#"<a self="closing"/>"#);
}

#[test]
fn write_event_basic() {
    let mut writer = Writer::new(Vec::new());
    writer
        .write_event(Event::Start(BytesStart::new("root")))
        .expect("start");
    writer
        .write_event(Event::Text(BytesText::new("a < b & c")))
        .expect("text");
    writer
        .write_event(Event::End(BytesEnd::new("root")))
        .expect("end");
    assert_eq!(
        writer.into_inner(),
        br#"<root>a &lt; b &amp; c</root>"#.to_vec(),
    );
}

#[test]
fn empty_element_slash_variants() {
    // Default: no space before slash.
    let mut writer = Writer::new(Vec::new());
    writer
        .create_element("tag")
        .with_attribute(("a", "1"))
        .write_empty()
        .expect("empty");
    assert_eq!(writer.into_inner(), br#"<tag a="1"/>"#.to_vec());

    // Configured: space before slash.
    let mut writer = Writer::new(Vec::new());
    writer.config_mut().add_space_before_slash_in_empty_elements = true;
    writer
        .create_element("tag")
        .with_attribute(("a", "1"))
        .write_empty()
        .expect("empty");
    assert_eq!(writer.into_inner(), br#"<tag a="1" />"#.to_vec());
}

#[test]
fn element_writer_text_cdata_pi_content() {
    let mut writer = Writer::new(Vec::new());
    writer
        .create_element("t")
        .write_text_content(BytesText::new("inside"))
        .expect("text");
    writer
        .create_element("c")
        .write_cdata_content(BytesCData::new(" raw "))
        .expect("cdata");
    writer
        .create_element("p")
        .write_pi_content(BytesPI::new("php echo 1"))
        .expect("pi");
    assert_eq!(
        std::str::from_utf8(&writer.into_inner()).expect("utf-8"),
        r#"<t>inside</t><c><![CDATA[ raw ]]></c><p><?php echo 1?></p>"#,
    );
}

#[test]
fn element_writer_inner_content() {
    let mut writer = Writer::new(Vec::new());
    writer
        .create_element("list")
        .write_inner_content(|writer| {
            for item in ["a", "b"] {
                writer
                    .create_element("item")
                    .write_text_content(BytesText::new(item))?;
            }
            Ok(())
        })
        .expect("inner");
    assert_eq!(
        std::str::from_utf8(&writer.into_inner()).expect("utf-8"),
        r#"<list><item>a</item><item>b</item></list>"#,
    );
}

#[test]
fn with_attributes_from_iterator() {
    let mut writer = Writer::new(Vec::new());
    writer
        .create_element("tag")
        .with_attributes([("a", "1"), ("b", "2")])
        .write_empty()
        .expect("empty");
    assert_eq!(writer.into_inner(), br#"<tag a="1" b="2"/>"#.to_vec());
}

#[test]
fn indentation_matches_quick_xml() {
    let mut writer = Writer::new_with_indent(Vec::new(), b' ', 4);
    writer
        .write_event(Event::Start(BytesStart::new("a")))
        .expect("a");
    writer
        .write_event(Event::Start(BytesStart::new("b")))
        .expect("b");
    writer
        .write_event(Event::Text(BytesText::new("text")))
        .expect("text");
    writer
        .write_event(Event::End(BytesEnd::new("b")))
        .expect("/b");
    writer
        .write_event(Event::End(BytesEnd::new("a")))
        .expect("/a");
    assert_eq!(
        std::str::from_utf8(&writer.into_inner()).expect("utf-8"),
        "<a>\n    <b>text</b>\n</a>",
    );
}

#[test]
fn indentation_nested_empty() {
    let mut writer = Writer::new_with_indent(Vec::new(), b' ', 4);
    writer
        .write_event(Event::Start(BytesStart::new("a")))
        .expect("a");
    writer
        .write_event(Event::Start(BytesStart::new("b")))
        .expect("b");
    writer
        .write_event(Event::Empty(BytesStart::new("c")))
        .expect("c");
    writer
        .write_event(Event::End(BytesEnd::new("b")))
        .expect("/b");
    writer
        .write_event(Event::End(BytesEnd::new("a")))
        .expect("/a");
    assert_eq!(
        std::str::from_utf8(&writer.into_inner()).expect("utf-8"),
        "<a>\n    <b>\n        <c/>\n    </b>\n</a>",
    );
}

#[test]
fn indentation_with_tabs() {
    let mut writer = Writer::new_with_indent(Vec::new(), b'\t', 1);
    writer
        .write_event(Event::Start(BytesStart::new("a")))
        .expect("a");
    writer
        .write_event(Event::Empty(BytesStart::new("b")))
        .expect("b");
    writer
        .write_event(Event::End(BytesEnd::new("a")))
        .expect("/a");
    assert_eq!(
        std::str::from_utf8(&writer.into_inner()).expect("utf-8"),
        "<a>\n\t<b/>\n</a>",
    );
}

#[test]
fn new_line_between_attributes() {
    let mut writer = Writer::new_with_indent(Vec::new(), b' ', 2);
    writer
        .create_element("element")
        .with_attribute(("first", "1"))
        .with_attribute(("second", "2"))
        .new_line()
        .with_attributes([("third", "3"), ("fourth", "4")])
        .write_empty()
        .expect("empty");
    assert_eq!(
        std::str::from_utf8(&writer.into_inner()).expect("utf-8"),
        "<element first=\"1\" second=\"2\"\n         third=\"3\" fourth=\"4\"/>",
    );
}

#[test]
fn write_bom_prepends_marker() {
    let mut writer = Writer::new(Vec::new());
    writer.write_bom().expect("bom");
    writer.create_element("a").write_empty().expect("empty");
    assert_eq!(
        std::str::from_utf8(&writer.into_inner()).expect("utf-8"),
        "\u{FEFF}<a/>",
    );
}

#[test]
fn write_indent_manual() {
    let mut writer = Writer::new_with_indent(Vec::new(), b' ', 2);
    writer
        .write_event(Event::Start(BytesStart::new("a")))
        .expect("a");
    // A start after text would not line-break automatically; force it manually.
    writer
        .write_event(Event::Text(BytesText::new("x")))
        .expect("text");
    writer.write_indent().expect("indent");
    writer
        .write_event(Event::End(BytesEnd::new("a")))
        .expect("/a");
    // `write_indent` emits the newline at the current (pre-shrink) level.
    assert_eq!(
        std::str::from_utf8(&writer.into_inner()).expect("utf-8"),
        "<a>x\n  </a>",
    );
}

#[test]
fn get_ref_get_mut_into_inner() {
    let mut writer = Writer::new(Vec::new());
    writer
        .write_event(Event::Empty(BytesStart::new("a")))
        .expect("a");
    assert_eq!(writer.get_ref(), b"<a/>");
    writer.get_mut().extend_from_slice(b"!");
    assert_eq!(writer.into_inner(), b"<a/>!".to_vec());
}

#[test]
fn round_trip_via_reader_and_writer() {
    // Read -> rename an element -> write, mirroring the crate-level example.
    let xml = r#"<this k="v"><child>text</child></this>"#;
    let mut reader = Reader::from_str(xml);
    let mut writer = Writer::new(Vec::new());
    loop {
        match reader.read_event().expect("event") {
            Event::Start(e) if e.name().as_ref() == b"this" => {
                let mut elem = BytesStart::new("that");
                elem.extend_attributes(e.attributes().map(|a| a.expect("attr")));
                writer.write_event(Event::Start(elem)).expect("start");
            }
            Event::End(e) if e.name().as_ref() == b"this" => {
                writer
                    .write_event(Event::End(BytesEnd::new("that")))
                    .expect("end");
            }
            Event::Eof => break,
            e => writer.write_event(e).expect("event"),
        }
    }
    assert_eq!(
        std::str::from_utf8(&writer.into_inner()).expect("utf-8"),
        r#"<that k="v"><child>text</child></that>"#,
    );
}
