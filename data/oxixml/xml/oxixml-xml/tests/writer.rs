//! Integration tests for the events-based [`Writer`].
//!
//! The event-serialization cases are ported from quick-xml 0.41 `tests/writer.rs`
//! (MIT/Apache-2.0, © the quick-xml authors), adapted to the `oxixml_xml` public
//! API and to the COOLJAPAN no-`unwrap()` policy (tests use `?`/`expect`). The
//! remaining cases are `oxixml`-specific: the [`ElementWriter`] builder surface,
//! writer accessors and the empty-element slash configuration.
//!
//! [`Writer`]: oxixml_xml::writer::Writer
//! [`ElementWriter`]: oxixml_xml::writer::ElementWriter

use std::io;

use oxixml_xml::events::Event::*;
use oxixml_xml::events::{
    BytesCData, BytesDecl, BytesEnd, BytesPI, BytesRef, BytesStart, BytesText,
};
use oxixml_xml::writer::Writer;

type TestResult = io::Result<()>;

/// Writes a single event with a plain (non-indenting) writer and returns the
/// produced document as a `String`.
fn write_one<'a, E: Into<oxixml_xml::events::Event<'a>>>(event: E) -> String {
    let mut writer = Writer::new(Vec::new());
    writer.write_event(event).expect("write event");
    String::from_utf8(writer.into_inner()).expect("utf-8 output")
}

mod declaration {
    use super::*;

    /// Written: version, encoding, standalone.
    #[test]
    fn full() {
        assert_eq!(
            write_one(Decl(BytesDecl::new("1.2", Some("utf-X"), Some("yo")))),
            r#"<?xml version="1.2" encoding="utf-X" standalone="yo"?>"#,
        );
    }

    /// Written: version, standalone.
    #[test]
    fn standalone() {
        assert_eq!(
            write_one(Decl(BytesDecl::new("1.2", None, Some("yo")))),
            r#"<?xml version="1.2" standalone="yo"?>"#,
        );
    }

    /// Written: version, encoding.
    #[test]
    fn encoding() {
        assert_eq!(
            write_one(Decl(BytesDecl::new("1.2", Some("utf-X"), None))),
            r#"<?xml version="1.2" encoding="utf-X"?>"#,
        );
    }

    /// Written: version.
    #[test]
    fn version() {
        assert_eq!(
            write_one(Decl(BytesDecl::new("1.2", None, None))),
            r#"<?xml version="1.2"?>"#,
        );
    }

    /// Ensures that empty XML-declaration attribute values are not a problem.
    /// An empty version is arguably an error, but nobody is expected to supply one.
    #[test]
    fn empty() {
        assert_eq!(
            write_one(Decl(BytesDecl::new("", Some(""), Some("")))),
            r#"<?xml version="" encoding="" standalone=""?>"#,
        );
    }
}

#[test]
fn pi() {
    assert_eq!(
        write_one(PI(BytesPI::new("xml-stylesheet href='theme.xls' "))),
        "<?xml-stylesheet href='theme.xls' ?>",
    );
}

#[test]
fn empty() {
    assert_eq!(
        write_one(Empty(
            BytesStart::new("game").with_attributes([("publisher", "Blizzard")])
        )),
        r#"<game publisher="Blizzard"/>"#,
    );
}

#[test]
fn start() {
    assert_eq!(
        write_one(Start(
            BytesStart::new("info").with_attributes([("genre", "RTS")])
        )),
        r#"<info genre="RTS">"#,
    );
}

#[test]
fn end() {
    assert_eq!(write_one(End(BytesEnd::new("info"))), "</info>");
}

#[test]
fn text() {
    assert_eq!(
        write_one(Text(BytesText::new(
            "Kerrigan & Raynor: The Z[erg] programming language",
        ))),
        "Kerrigan &amp; Raynor: The Z[erg] programming language",
    );
}

#[test]
fn cdata() {
    assert_eq!(
        write_one(CData(BytesCData::new(
            "Kerrigan & Raynor: The Z[erg] programming language",
        ))),
        "<![CDATA[Kerrigan & Raynor: The Z[erg] programming language]]>",
    );
}

#[test]
fn comment() {
    assert_eq!(
        write_one(Comment(BytesText::from_escaped(
            "Kerrigan & Raynor: The Z[erg] programming language",
        ))),
        "<!--Kerrigan & Raynor: The Z[erg] programming language-->",
    );
}

#[test]
fn doctype() {
    assert_eq!(
        write_one(DocType(BytesText::new("some DTD here..."))),
        "<!DOCTYPE some DTD here...>",
    );
}

#[test]
fn general_ref() {
    assert_eq!(write_one(GeneralRef(BytesRef::new("entity"))), "&entity;");
}

#[test]
fn eof() {
    assert_eq!(write_one(Eof), "");
}

/// A borrowed event (`&Event`) must serialize exactly like an owned one.
#[test]
fn write_borrowed_event() {
    let event = Start(BytesStart::new("info").with_attributes([("genre", "RTS")]));
    let mut writer = Writer::new(Vec::new());
    writer.write_event(event.borrow()).expect("write borrowed");
    writer.write_event(event).expect("write owned");
    assert_eq!(
        String::from_utf8(writer.into_inner()).expect("utf-8 output"),
        r#"<info genre="RTS"><info genre="RTS">"#,
    );
}

#[test]
fn write_bom_prepends_marker() -> TestResult {
    let mut writer = Writer::new(Vec::new());
    writer.write_bom()?;
    writer.create_element("a").write_empty()?;
    assert_eq!(
        String::from_utf8(writer.into_inner()).expect("utf-8 output"),
        "\u{FEFF}<a/>",
    );
    Ok(())
}

#[test]
fn empty_element_slash_variants() -> TestResult {
    // Default: no space before the slash.
    let mut writer = Writer::new(Vec::new());
    writer
        .create_element("tag")
        .with_attribute(("a", "1"))
        .write_empty()?;
    assert_eq!(writer.into_inner(), br#"<tag a="1"/>"#.to_vec());

    // Configured: a space before the slash, as the W3C XHTML guidelines suggest.
    let mut writer = Writer::new(Vec::new());
    writer.config_mut().add_space_before_slash_in_empty_elements = true;
    writer
        .create_element("tag")
        .with_attribute(("a", "1"))
        .write_empty()?;
    assert_eq!(writer.into_inner(), br#"<tag a="1" />"#.to_vec());
    Ok(())
}

mod element_writer {
    use super::*;

    #[test]
    fn write_empty_without_attributes() -> TestResult {
        let mut writer = Writer::new(Vec::new());
        writer.create_element("tag").write_empty()?;
        assert_eq!(writer.into_inner(), b"<tag/>".to_vec());
        Ok(())
    }

    #[test]
    fn with_attribute_and_with_attributes() -> TestResult {
        let mut writer = Writer::new(Vec::new());
        writer
            .create_element("tag")
            .with_attribute(("a", "1"))
            .with_attributes([("b", "2"), ("c", "3")])
            .write_empty()?;
        assert_eq!(writer.into_inner(), br#"<tag a="1" b="2" c="3"/>"#.to_vec());
        Ok(())
    }

    /// `with_attributes` over an empty iterator must be a no-op.
    #[test]
    fn with_attributes_empty_iterator() -> TestResult {
        let mut writer = Writer::new(Vec::new());
        let none: [(&str, &str); 0] = [];
        writer
            .create_element("tag")
            .with_attributes(none)
            .write_empty()?;
        assert_eq!(writer.into_inner(), b"<tag/>".to_vec());
        Ok(())
    }

    /// Attribute values are escaped when pushed through the builder.
    #[test]
    fn attribute_values_are_escaped() -> TestResult {
        let mut writer = Writer::new(Vec::new());
        writer
            .create_element("tag")
            .with_attribute(("a", r#"<&">"#))
            .write_empty()?;
        assert_eq!(
            String::from_utf8(writer.into_inner()).expect("utf-8 output"),
            r#"<tag a="&lt;&amp;&quot;&gt;"/>"#,
        );
        Ok(())
    }

    #[test]
    fn write_text_content() -> TestResult {
        let mut writer = Writer::new(Vec::new());
        writer
            .create_element("t")
            .write_text_content(BytesText::new("a < b"))?;
        assert_eq!(
            String::from_utf8(writer.into_inner()).expect("utf-8 output"),
            "<t>a &lt; b</t>",
        );
        Ok(())
    }

    #[test]
    fn write_cdata_content() -> TestResult {
        let mut writer = Writer::new(Vec::new());
        writer
            .create_element("c")
            .write_cdata_content(BytesCData::new(" raw <b> & "))?;
        assert_eq!(
            String::from_utf8(writer.into_inner()).expect("utf-8 output"),
            "<c><![CDATA[ raw <b> & ]]></c>",
        );
        Ok(())
    }

    #[test]
    fn write_pi_content() -> TestResult {
        let mut writer = Writer::new(Vec::new());
        writer
            .create_element("p")
            .write_pi_content(BytesPI::new("php echo 1"))?;
        assert_eq!(
            String::from_utf8(writer.into_inner()).expect("utf-8 output"),
            "<p><?php echo 1?></p>",
        );
        Ok(())
    }

    #[test]
    fn write_inner_content() -> TestResult {
        let mut writer = Writer::new(Vec::new());
        writer
            .create_element("list")
            .write_inner_content(|writer| {
                for (quantity, item) in ["apple", "orange"].iter().enumerate() {
                    writer
                        .create_element("fruit")
                        .with_attribute(("quantity", quantity.to_string().as_str()))
                        .write_text_content(BytesText::new(item))?;
                }
                Ok(())
            })?;
        assert_eq!(
            String::from_utf8(writer.into_inner()).expect("utf-8 output"),
            r#"<list><fruit quantity="0">apple</fruit><fruit quantity="1">orange</fruit></list>"#,
        );
        Ok(())
    }

    /// An error raised inside the closure is propagated out of
    /// `write_inner_content`, and the start tag has already been emitted.
    #[test]
    fn write_inner_content_propagates_errors() {
        let mut writer = Writer::new(Vec::new());
        let result = writer
            .create_element("list")
            .write_inner_content(|_| Err(io::Error::other("boom")));
        let error = result.err().expect("closure error is propagated");
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert_eq!(writer.into_inner(), b"<list>".to_vec());
    }

    /// Every terminal builder method returns the writer, so calls can be chained.
    #[test]
    fn builder_returns_the_writer() -> TestResult {
        let mut writer = Writer::new(Vec::new());
        writer
            .create_element("a")
            .write_empty()?
            .create_element("b")
            .write_text_content(BytesText::new("x"))?
            .create_element("c")
            .write_empty()?;
        assert_eq!(
            String::from_utf8(writer.into_inner()).expect("utf-8 output"),
            "<a/><b>x</b><c/>",
        );
        Ok(())
    }
}

#[test]
fn get_ref_get_mut_into_inner() -> TestResult {
    let mut writer = Writer::new(Vec::new());
    writer.write_event(Empty(BytesStart::new("a")))?;
    assert_eq!(writer.get_ref(), b"<a/>");
    writer.get_mut().extend_from_slice(b"!");
    assert_eq!(writer.into_inner(), b"<a/>!".to_vec());
    Ok(())
}

/// The writer may borrow its sink, which is the shape used by most of the
/// indentation tests.
#[test]
fn writes_through_a_borrowed_sink() -> TestResult {
    let mut buffer = Vec::new();
    {
        let mut writer = Writer::new(&mut buffer);
        writer.write_event(Empty(BytesStart::new("a")))?;
    }
    assert_eq!(buffer, b"<a/>".to_vec());
    Ok(())
}

/// I/O errors from the sink surface unchanged.
#[test]
fn io_errors_are_propagated() {
    /// A sink that fails on every write.
    struct Failing;

    impl io::Write for Failing {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "nope"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut writer = Writer::new(Failing);
    let error = writer
        .write_event(Empty(BytesStart::new("a")))
        .expect_err("sink error is propagated");
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
}
