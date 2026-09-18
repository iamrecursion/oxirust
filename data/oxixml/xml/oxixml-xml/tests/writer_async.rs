//! Integration tests for the `async-tokio` [`Writer`] methods.
//!
//! Exercises a full async round trip (parse with
//! [`Reader::read_event_into_async`], serialize with
//! [`Writer::write_event_async`]) and cross-checks that the async writer
//! produces byte-identical output to the synchronous [`Writer::write_event`]
//! for the same event stream, including with indentation and the
//! [`ElementWriter`] builder's async content methods.
//!
//! [`Reader::read_event_into_async`]: oxixml_xml::reader::Reader::read_event_into_async
//! [`Writer::write_event_async`]: oxixml_xml::writer::Writer::write_event_async

use oxixml_xml::errors::Error;
use oxixml_xml::events::{BytesCData, BytesEnd, BytesStart, BytesText, Event};
use oxixml_xml::reader::Reader;
use oxixml_xml::writer::Writer;

/// Reads every event of `xml` into owned events, excluding the final `Eof`.
fn read_all(xml: &str) -> Result<Vec<Event<'static>>, Error> {
    let mut reader = Reader::from_str(xml);
    let mut events = Vec::new();
    loop {
        match reader.read_event()? {
            Event::Eof => break,
            e => events.push(e.into_owned()),
        }
    }
    Ok(events)
}

/// Writes a stream of events synchronously and returns the produced document.
fn write_all_sync(events: &[Event<'_>], indent: Option<(u8, usize)>) -> String {
    let mut writer = match indent {
        Some((ch, size)) => Writer::new_with_indent(Vec::new(), ch, size),
        None => Writer::new(Vec::new()),
    };
    for event in events {
        writer.write_event(event.borrow()).expect("write event");
    }
    String::from_utf8(writer.into_inner()).expect("utf-8 output")
}

/// Writes a stream of events asynchronously and returns the produced document.
async fn write_all_async(events: &[Event<'_>], indent: Option<(u8, usize)>) -> String {
    let mut writer = match indent {
        Some((ch, size)) => Writer::new_with_indent(Vec::new(), ch, size),
        None => Writer::new(Vec::new()),
    };
    for event in events {
        writer
            .write_event_async(event.borrow())
            .await
            .expect("write event async");
    }
    String::from_utf8(writer.into_inner()).expect("utf-8 output")
}

/// Reads every event of `xml` asynchronously via a buffered async source, into
/// owned events, excluding the final `Eof`.
async fn read_all_async(xml: &str) -> Result<Vec<Event<'static>>, Error> {
    let mut reader = Reader::from_reader(xml.as_bytes());
    let mut buf = Vec::new();
    let mut events = Vec::new();
    loop {
        match reader.read_event_into_async(&mut buf).await? {
            Event::Eof => break,
            e => events.push(e.into_owned()),
        }
        buf.clear();
    }
    Ok(events)
}

fn corpus() -> Vec<&'static str> {
    vec![
        r#"<root>text</root>"#,
        r#"<a k1="v1" k2="v2"><b/><c>x</c></a>"#,
        r#"<x:a xmlns:x="urn:x"><x:b>t</x:b></x:a>"#,
        r#"<a><!--a comment--></a>"#,
        r#"<a><![CDATA[ raw <b> & ]]></a>"#,
        r#"<a>one &amp; two &lt; three</a>"#,
        r#"<?xml version="1.0" encoding="UTF-8"?><root/>"#,
        r#"<a><?target instruction?></a>"#,
        r#"<a self="closing"/>"#,
    ]
}

#[tokio::test]
async fn async_round_trip_matches_sync_for_corpus() -> Result<(), Error> {
    for doc in corpus() {
        let events = read_all(doc)?;

        let sync_out = write_all_sync(&events, None);
        let async_out = write_all_async(&events, None).await;
        assert_eq!(async_out, sync_out, "mismatch for doc: {doc:?}");

        let sync_indented = write_all_sync(&events, Some((b' ', 2)));
        let async_indented = write_all_async(&events, Some((b' ', 2))).await;
        assert_eq!(
            async_indented, sync_indented,
            "mismatch (indented) for doc: {doc:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn async_read_then_async_write_round_trips() -> Result<(), Error> {
    for doc in corpus() {
        let events = read_all_async(doc).await?;
        let written = write_all_async(&events, None).await;
        assert_eq!(written, doc, "mismatch for doc: {doc:?}");

        // Re-parsing the async-written output synchronously must reproduce the
        // same event stream, closing the loop between the async reader and the
        // async writer through the (already exhaustively tested) sync reader.
        let reparsed = read_all(&written)?;
        assert_eq!(events, reparsed, "event stream changed after round trip");
    }
    Ok(())
}

#[tokio::test]
async fn element_writer_async_content_methods() -> Result<(), Error> {
    let mut writer = Writer::new(Vec::new());
    writer
        .create_element("t")
        .write_text_content_async(BytesText::new("inside"))
        .await?;
    writer
        .create_element("c")
        .write_cdata_content_async(BytesCData::new(" raw "))
        .await?;
    writer.create_element("e").write_empty_async().await?;
    writer
        .create_element("outer")
        .write_inner_content_async::<_, _, Error>(|w| async move {
            w.create_element("inner").write_empty_async().await
        })
        .await?;

    assert_eq!(
        std::str::from_utf8(&writer.into_inner()).expect("utf-8"),
        r#"<t>inside</t><c><![CDATA[ raw ]]></c><e/><outer><inner/></outer>"#,
    );
    Ok(())
}

#[tokio::test]
async fn write_indent_async_matches_sync() -> Result<(), Error> {
    let mut sync_writer = Writer::new_with_indent(Vec::new(), b' ', 2);
    sync_writer
        .write_event(Event::Start(BytesStart::new("a")))
        .expect("start");
    sync_writer
        .write_event(Event::Text(BytesText::new("x")))
        .expect("text");
    sync_writer.write_indent().expect("indent");
    sync_writer
        .write_event(Event::End(BytesEnd::new("a")))
        .expect("end");

    let mut async_writer = Writer::new_with_indent(Vec::new(), b' ', 2);
    async_writer
        .write_event_async(Event::Start(BytesStart::new("a")))
        .await?;
    async_writer
        .write_event_async(Event::Text(BytesText::new("x")))
        .await?;
    async_writer.write_indent_async().await?;
    async_writer
        .write_event_async(Event::End(BytesEnd::new("a")))
        .await?;

    assert_eq!(async_writer.into_inner(), sync_writer.into_inner());
    Ok(())
}
