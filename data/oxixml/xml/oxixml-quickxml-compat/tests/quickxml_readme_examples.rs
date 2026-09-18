//! Compiles and runs the `quick-xml` 0.41 `README.md` examples verbatim
//! against `oxixml-quickxml-compat`, proving that a downstream crate written
//! against `quick_xml` keeps working unmodified after aliasing its dependency
//! to `oxixml-quickxml-compat` (`quick_xml = { package = "oxixml-quickxml-compat" }`).
//!
//! Only the `use` paths are rewritten (`use quick_xml::...` becomes
//! `use oxixml_quickxml_compat as quick_xml; use quick_xml::...`), matching
//! exactly how a real downstream `Cargo.toml` alias would work: every other
//! line is untouched.

use oxixml_quickxml_compat as quick_xml;

/// The README "Reader" example, verbatim.
#[test]
fn readme_reader_example() {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    let xml = r#"<tag1 att1 = "test">
                    <tag2><!--Test comment-->Test</tag2>
                    <tag2>Test 2</tag2>
                 </tag1>"#;
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut count = 0;
    let mut txt = Vec::new();
    let mut buf = Vec::new();

    // The `Reader` does not implement `Iterator` because it outputs borrowed data (`Cow`s)
    loop {
        // NOTE: this is the generic case when we don't know about the input BufRead.
        // when the input is a &str or a &[u8], we don't actually need to use another
        // buffer, we could directly call `reader.read_event()`
        match reader.read_event_into(&mut buf) {
            Err(e) => panic!("Error at position {}: {:?}", reader.error_position(), e),
            // exits the loop when reaching end of file
            Ok(Event::Eof) => break,

            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"tag1" => println!(
                    "attributes values: {:?}",
                    e.attributes()
                        .map(|a| a.expect("attribute").value)
                        .collect::<Vec<_>>()
                ),
                b"tag2" => count += 1,
                _ => (),
            },
            Ok(Event::Text(e)) => txt.push(e.decode().expect("utf-8").into_owned()),

            // There are several other `Event`s we do not consider here
            _ => (),
        }
        // if we don't keep a borrow elsewhere, we can clear the buffer to keep memory usage low
        buf.clear();
    }

    assert_eq!(count, 2);
    assert_eq!(txt, vec!["Test".to_string(), "Test 2".to_string()]);
}

/// The README "Writer" example, verbatim.
#[test]
fn readme_writer_example() {
    use quick_xml::events::{BytesEnd, BytesStart, Event};
    use quick_xml::reader::Reader;
    use quick_xml::writer::Writer;
    use std::io::Cursor;

    let xml = r#"<this_tag k1="v1" k2="v2"><child>text</child></this_tag>"#;
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut writer = Writer::new(Cursor::new(Vec::new()));
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) if e.name().as_ref() == b"this_tag" => {
                // creates a new element ... alternatively we could reuse `e` by calling
                // `e.into_owned()`
                let mut elem = BytesStart::new("my_elem");

                // collect existing attributes
                elem.extend_attributes(e.attributes().map(|attr| attr.expect("attribute")));

                // copy existing attributes, adds a new my-key="some value" attribute
                elem.push_attribute(("my-key", "some value"));

                // writes the event to the writer
                assert!(writer.write_event(Event::Start(elem)).is_ok());
            }
            Ok(Event::End(e)) if e.name().as_ref() == b"this_tag" => {
                assert!(
                    writer
                        .write_event(Event::End(BytesEnd::new("my_elem")))
                        .is_ok()
                );
            }
            Ok(Event::Eof) => break,
            // we can either move or borrow the event to write, depending on your use-case
            Ok(e) => assert!(writer.write_event(e).is_ok()),
            Err(e) => panic!("Error at position {}: {:?}", reader.error_position(), e),
        }
    }

    let result = writer.into_inner().into_inner();
    let expected = r#"<my_elem k1="v1" k2="v2" my-key="some value"><child>text</child></my_elem>"#;
    assert_eq!(result, expected.as_bytes());
}
