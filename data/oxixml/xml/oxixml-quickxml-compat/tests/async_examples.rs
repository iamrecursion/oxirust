//! Compiles and runs the `quick-xml` 0.41 `async-tokio` rustdoc examples
//! against `oxixml-quickxml-compat`, proving async reader/writer
//! path-compatibility. Requires the `async-tokio` feature.

use oxixml_quickxml_compat as quick_xml;

/// `quick_xml::reader::Reader::read_event_into_async` doc example.
#[tokio::test]
async fn reader_read_event_into_async() {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    // This explicitly uses `from_reader("...".as_bytes())` to use a buffered
    // reader instead of relying on the zero-copy optimizations for reading
    // from byte slices, which provides the sync interface anyway.
    let mut reader = Reader::from_reader(
        r#"
        <tag1 att1 = "test">
           <tag2><!--Test comment-->Test</tag2>
           <tag2>Test 2</tag2>
        </tag1>
    "#
        .as_bytes(),
    );
    reader.config_mut().trim_text(true);

    let mut count = 0;
    let mut buf = Vec::new();
    let mut txt = Vec::new();
    loop {
        match reader.read_event_into_async(&mut buf).await {
            Ok(Event::Start(_)) => count += 1,
            Ok(Event::Text(e)) => txt.push(e.decode().expect("utf-8").into_owned()),
            Err(e) => panic!("Error at position {}: {:?}", reader.error_position(), e),
            Ok(Event::Eof) => break,
            _ => (),
        }
        buf.clear();
    }
    assert_eq!(count, 3);
    assert_eq!(txt, vec!["Test".to_string(), "Test 2".to_string()]);
}

/// `quick_xml::reader::Reader::read_to_end_into_async` doc example: skips
/// nested content after a `Start` event, correctly handling nested elements
/// that share the outer element's name.
#[tokio::test]
async fn reader_read_to_end_into_async_skips_nested_content() {
    use quick_xml::events::{BytesStart, Event};
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_reader(
        r#"
        <outer>
            <inner>
                <inner></inner>
                <inner/>
                <outer></outer>
                <outer/>
            </inner>
        </outer>
    "#
        .as_bytes(),
    );
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();

    let start = BytesStart::new("outer");
    let end = start.to_end().into_owned();

    // First, we read a start event...
    assert_eq!(
        reader.read_event_into_async(&mut buf).await.expect("start"),
        Event::Start(start)
    );

    // ...then, we could skip all events to the corresponding end event.
    // This call will correctly handle nested <outer> elements.
    reader
        .read_to_end_into_async(end.name(), &mut buf)
        .await
        .expect("skip to end");

    // At the end we should get an Eof event, because we ate the whole XML.
    assert_eq!(
        reader.read_event_into_async(&mut buf).await.expect("eof"),
        Event::Eof
    );
}

/// Async writer round trip, mirroring the sync README writer example but
/// driven through `Writer::write_event_async`.
#[tokio::test]
async fn writer_write_event_async_round_trip() {
    use quick_xml::events::{BytesEnd, BytesStart, Event};
    use quick_xml::reader::Reader;
    use quick_xml::writer::Writer;

    let xml = r#"<this_tag k1="v1" k2="v2"><child>text</child></this_tag>"#;
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut writer = Writer::new(Vec::new());
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) if e.name().as_ref() == b"this_tag" => {
                let mut elem = BytesStart::new("my_elem");
                elem.extend_attributes(e.attributes().map(|attr| attr.expect("attribute")));
                elem.push_attribute(("my-key", "some value"));
                writer
                    .write_event_async(Event::Start(elem))
                    .await
                    .expect("write start");
            }
            Ok(Event::End(e)) if e.name().as_ref() == b"this_tag" => {
                writer
                    .write_event_async(Event::End(BytesEnd::new("my_elem")))
                    .await
                    .expect("write end");
            }
            Ok(Event::Eof) => break,
            Ok(e) => writer.write_event_async(e).await.expect("write event"),
            Err(e) => panic!("Error at position {}: {:?}", reader.error_position(), e),
        }
    }

    let result = writer.into_inner();
    let expected = r#"<my_elem k1="v1" k2="v2" my-key="some value"><child>text</child></my_elem>"#;
    assert_eq!(result, expected.as_bytes());
}
