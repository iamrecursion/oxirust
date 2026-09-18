//! Miscellaneous deserializer tests that do not belong to a larger themed group.

use super::*;

#[test]
fn borrowing_reader_parity() {
    let s = r#"
            <item name="hello" source="world.rs">Some text</item>
            <item2/>
            <item3 value="world" />
        "#;

    let mut reader1 = IoReader {
        reader: NsReader::from_reader(s.as_bytes()),
        buf: Vec::new(),
        version: XmlVersion::Implicit1_0,
    };
    let mut reader2 = SliceReader {
        reader: NsReader::from_str(s),
        version: XmlVersion::Implicit1_0,
    };

    loop {
        let event1 = reader1.next().expect("should be Ok");
        let event2 = reader2.next().expect("should be Ok");

        if let (PayloadEvent::Eof, PayloadEvent::Eof) = (&event1, &event2) {
            break;
        }

        assert_eq!(event1, event2);
    }
}

#[test]
fn borrowing_reader_events() {
    let s = r#"
            <item name="hello" source="world.rs">Some text</item>
            <item2></item2>
            <item3/>
            <item4 value="world" />
        "#;

    let mut reader = SliceReader {
        reader: NsReader::from_str(s),
        version: XmlVersion::Implicit1_0,
    };

    let config = reader.reader.config_mut();
    config.expand_empty_elements = true;

    let mut events = Vec::new();

    loop {
        let event = reader.next().expect("should be Ok");
        if let PayloadEvent::Eof = event {
            break;
        }
        events.push(event);
    }

    use crate::de::PayloadEvent::*;

    assert_eq!(
        events,
        vec![
            Text(BytesText::from_escaped("\n            ")),
            Start(BytesStart::from_content(
                r#"item name="hello" source="world.rs""#,
                4
            )),
            Text(BytesText::from_escaped("Some text")),
            End(BytesEnd::new("item")),
            Text(BytesText::from_escaped("\n            ")),
            Start(BytesStart::from_content("item2", 5)),
            End(BytesEnd::new("item2")),
            Text(BytesText::from_escaped("\n            ")),
            Start(BytesStart::from_content("item3", 5)),
            End(BytesEnd::new("item3")),
            Text(BytesText::from_escaped("\n            ")),
            Start(BytesStart::from_content(r#"item4 value="world" "#, 5)),
            End(BytesEnd::new("item4")),
            Text(BytesText::from_escaped("\n        ")),
        ]
    )
}

/// Ensures, that [`Deserializer::read_string()`] never can get an `End` event,
/// because parser reports error early
#[test]
fn read_string() {
    match from_str::<String>(r#"</root>"#) {
        Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
            assert_eq!(cause, IllFormedError::UnmatchedEndTag("root".into()));
        }
        x => panic!(
            "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
            x
        ),
    }

    let s: String = from_str(r#"<root></root>"#).expect("should be Ok");
    assert_eq!(s, "");

    match from_str::<String>(r#"<root></other>"#) {
        Err(DeError::InvalidXml(Error::IllFormed(cause))) => assert_eq!(
            cause,
            IllFormedError::MismatchedEndTag {
                expected: "root".into(),
                found: "other".into(),
            }
        ),
        x => panic!("Expected `Err(InvalidXml(IllFormed(_))`, but got `{:?}`", x),
    }
}

/// Pins the error `overview.md`'s "Basics" table promises for the three ways
/// borrowing a `&str` can fail: an entity reference, a character reference, or
/// a mixed text/CDATA run all force an owned `String` internally, and handing
/// that to a visitor that only accepts `visit_borrowed_str` fails with the
/// ordinary `serde::de::Error::invalid_type` message (surfaced here as
/// `DeError::Custom`) rather than a panic or a silent owned fallback.
#[test]
fn borrow_str_fails_ordinarily_when_content_forces_owned() {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct Entity<'a> {
        #[serde(rename = "$text")]
        #[allow(dead_code)]
        text: &'a str,
    }
    match from_str::<Entity>("<root>a&amp;b</root>") {
        Err(DeError::Custom(msg)) => assert!(msg.contains("expected a borrowed string")),
        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
    }

    #[derive(Debug, Deserialize)]
    struct CharRef<'a> {
        #[serde(rename = "$text")]
        #[allow(dead_code)]
        text: &'a str,
    }
    match from_str::<CharRef>("<root>&#65;</root>") {
        Err(DeError::Custom(msg)) => assert!(msg.contains("expected a borrowed string")),
        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
    }

    #[derive(Debug, Deserialize)]
    struct MixedRun<'a> {
        #[serde(rename = "$text")]
        #[allow(dead_code)]
        text: &'a str,
    }
    match from_str::<MixedRun>("<root>text<![CDATA[cdata]]></root>") {
        Err(DeError::Custom(msg)) => assert!(msg.contains("expected a borrowed string")),
        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
    }

    // Contrast: a single Text-only run with no entities does borrow.
    #[derive(Debug, Deserialize)]
    struct PlainText<'a> {
        #[serde(rename = "$text")]
        text: &'a str,
    }
    let ok: PlainText = from_str("<root>plain</root>").expect("should be Ok");
    assert_eq!(ok.text, "plain");
}

/// Pins the error an enum with no matching variant and no `#[serde(other)]`
/// fallback produces: an ordinary `serde::de::Error::unknown_variant` message
/// (surfaced as `DeError::Custom`), exactly as the "NOTE" in `overview.md`'s
/// tagged-enum section promises must be documented.
#[test]
fn enum_without_other_variant_rejects_unknown_tag() {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum AnyName {
        One,
    }
    match from_str::<AnyName>(r#"<two field2="x">...</two>"#) {
        Err(DeError::Custom(msg)) => {
            assert!(msg.contains("unknown variant"));
            assert!(msg.contains("two"));
        }
        x => panic!("Expected `Err(Custom(_))`, but got `{:?}`", x),
    }
}
