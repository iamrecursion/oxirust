//! Integration tests pinning the pretty-printing output of [`Writer`] to the
//! byte-exact behaviour of quick-xml 0.41.
//!
//! Ported from quick-xml 0.41 `tests/writer-indentation.rs` (MIT/Apache-2.0,
//! © the quick-xml authors), adapted to the `oxixml_xml` public API and to the
//! COOLJAPAN no-`unwrap()` policy (tests use `?`/`expect`). The `serializable`
//! case is intentionally absent — the serializer lands in a later pipeline stage.
//!
//! [`Writer`]: oxixml_xml::writer::Writer

use std::io;

use oxixml_xml::events::{BytesCData, BytesEnd, BytesStart, BytesText, Event};
use oxixml_xml::writer::Writer;

type TestResult = io::Result<()>;

/// Decodes the accumulated output of a borrowed buffer.
fn utf8(buffer: &[u8]) -> &str {
    std::str::from_utf8(buffer).expect("utf-8 output")
}

#[test]
fn self_closed() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    let tag = BytesStart::new("self-closed")
        .with_attributes(vec![("attr1", "value1"), ("attr2", "value2")]);
    writer.write_event(Event::Empty(tag))?;

    assert_eq!(
        utf8(&buffer),
        r#"<self-closed attr1="value1" attr2="value2"/>"#
    );
    Ok(())
}

#[test]
fn empty_paired() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    let start =
        BytesStart::new("paired").with_attributes(vec![("attr1", "value1"), ("attr2", "value2")]);
    let end = start.to_end();
    writer.write_event(Event::Start(start.clone()))?;
    writer.write_event(Event::End(end))?;

    assert_eq!(
        utf8(&buffer),
        r#"<paired attr1="value1" attr2="value2">
</paired>"#
    );
    Ok(())
}

#[test]
fn paired_with_inner() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    let start =
        BytesStart::new("paired").with_attributes(vec![("attr1", "value1"), ("attr2", "value2")]);
    let end = start.to_end();
    let inner = BytesStart::new("inner");

    writer.write_event(Event::Start(start.clone()))?;
    writer.write_event(Event::Empty(inner))?;
    writer.write_event(Event::End(end))?;

    assert_eq!(
        utf8(&buffer),
        r#"<paired attr1="value1" attr2="value2">
    <inner/>
</paired>"#
    );
    Ok(())
}

#[test]
fn paired_with_text() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    let start =
        BytesStart::new("paired").with_attributes(vec![("attr1", "value1"), ("attr2", "value2")]);
    let end = start.to_end();

    writer.write_event(Event::Start(start.clone()))?;
    writer.write_event(Event::Text(BytesText::new("text")))?;
    writer.write_event(Event::End(end))?;

    assert_eq!(
        utf8(&buffer),
        r#"<paired attr1="value1" attr2="value2">text</paired>"#
    );
    Ok(())
}

/// A CDATA section behaves like text: it stays on the current line and the
/// following end tag is not line-broken.
#[test]
fn paired_with_cdata() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    let start = BytesStart::new("paired");
    let end = start.to_end();

    writer.write_event(Event::Start(start.clone()))?;
    writer.write_event(Event::CData(BytesCData::new("raw & data")))?;
    writer.write_event(Event::End(end))?;

    assert_eq!(utf8(&buffer), "<paired><![CDATA[raw & data]]></paired>");
    Ok(())
}

#[test]
fn mixed_content() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    let start =
        BytesStart::new("paired").with_attributes(vec![("attr1", "value1"), ("attr2", "value2")]);
    let end = start.to_end();

    writer.write_event(Event::Start(start.clone()))?;
    writer.write_event(Event::Text(BytesText::new("text")))?;
    writer.write_event(Event::Empty(BytesStart::new("inner")))?;
    writer.write_event(Event::End(end))?;

    assert_eq!(
        utf8(&buffer),
        r#"<paired attr1="value1" attr2="value2">text<inner/>
</paired>"#
    );
    Ok(())
}

#[test]
fn nested() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    let start =
        BytesStart::new("paired").with_attributes(vec![("attr1", "value1"), ("attr2", "value2")]);
    let end = start.to_end();

    writer.write_event(Event::Start(start.clone()))?;
    writer.write_event(Event::Start(start.clone()))?;
    writer.write_event(Event::Empty(BytesStart::new("inner")))?;
    writer.write_event(Event::End(end.clone()))?;
    writer.write_event(Event::End(end))?;

    assert_eq!(
        utf8(&buffer),
        r#"<paired attr1="value1" attr2="value2">
    <paired attr1="value1" attr2="value2">
        <inner/>
    </paired>
</paired>"#
    );
    Ok(())
}

/// Tabs work exactly like spaces; only the fill byte changes.
#[test]
fn nested_with_tabs() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b'\t', 1);

    writer.write_event(Event::Start(BytesStart::new("a")))?;
    writer.write_event(Event::Start(BytesStart::new("b")))?;
    writer.write_event(Event::Empty(BytesStart::new("c")))?;
    writer.write_event(Event::End(BytesEnd::new("b")))?;
    writer.write_event(Event::End(BytesEnd::new("a")))?;

    assert_eq!(utf8(&buffer), "<a>\n\t<b>\n\t\t<c/>\n\t</b>\n</a>");
    Ok(())
}

/// Comments, processing instructions, declarations and doctypes all line-break
/// like element markup.
#[test]
fn non_element_markup_is_indented() -> TestResult {
    use oxixml_xml::events::{BytesDecl, BytesPI};

    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 2);

    writer.write_event(Event::Decl(BytesDecl::new("1.0", None, None)))?;
    writer.write_event(Event::DocType(BytesText::new("root")))?;
    writer.write_event(Event::Start(BytesStart::new("root")))?;
    writer.write_event(Event::Comment(BytesText::from_escaped("note")))?;
    writer.write_event(Event::PI(BytesPI::new("target data")))?;
    writer.write_event(Event::End(BytesEnd::new("root")))?;

    assert_eq!(
        utf8(&buffer),
        "<?xml version=\"1.0\"?>\n<!DOCTYPE root>\n<root>\n  <!--note-->\n  <?target data?>\n</root>",
    );
    Ok(())
}

/// `shrink` saturates at zero, so an unbalanced end tag cannot underflow the
/// indentation level.
#[test]
fn unbalanced_end_does_not_underflow() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    writer.write_event(Event::End(BytesEnd::new("a")))?;
    writer.write_event(Event::End(BytesEnd::new("b")))?;
    writer.write_event(Event::Empty(BytesStart::new("c")))?;

    assert_eq!(utf8(&buffer), "</a>\n</b>\n<c/>");
    Ok(())
}

/// `write_indent` emits a line break and the *current* indent, which is what the
/// automatic heuristic would otherwise skip after a text event.
#[test]
fn manual_write_indent() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 2);

    writer.write_event(Event::Start(BytesStart::new("a")))?;
    writer.write_event(Event::Text(BytesText::new("x")))?;
    writer.write_indent()?;
    writer.write_event(Event::Empty(BytesStart::new("b")))?;
    writer.write_event(Event::End(BytesEnd::new("a")))?;

    assert_eq!(utf8(&buffer), "<a>x\n  <b/>\n</a>");
    Ok(())
}

/// `write_indent` is a no-op on a writer created without indentation.
#[test]
fn manual_write_indent_without_indentation() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new(&mut buffer);

    writer.write_event(Event::Start(BytesStart::new("a")))?;
    writer.write_indent()?;
    writer.write_event(Event::End(BytesEnd::new("a")))?;

    assert_eq!(utf8(&buffer), "<a></a>");
    Ok(())
}

#[test]
fn element_writer_empty() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    writer
        .create_element("empty")
        .with_attribute(("attr1", "value1"))
        .with_attribute(("attr2", "value2"))
        .write_empty()?;

    assert_eq!(utf8(&buffer), r#"<empty attr1="value1" attr2="value2"/>"#);
    Ok(())
}

#[test]
fn element_writer_text() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    writer
        .create_element("paired")
        .with_attribute(("attr1", "value1"))
        .with_attribute(("attr2", "value2"))
        .write_text_content(BytesText::new("text"))?;

    assert_eq!(
        utf8(&buffer),
        r#"<paired attr1="value1" attr2="value2">text</paired>"#
    );
    Ok(())
}

#[test]
fn element_writer_nested() -> TestResult {
    let mut buffer = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

    writer
        .create_element("outer")
        .with_attribute(("attr1", "value1"))
        .with_attribute(("attr2", "value2"))
        .write_inner_content(|writer| {
            for (quantity, item) in ["apple", "orange", "banana"].iter().enumerate() {
                writer
                    .create_element("fruit")
                    .with_attribute(("quantity", quantity.to_string().as_str()))
                    .write_text_content(BytesText::new(item))?;
            }
            writer
                .create_element("inner")
                .write_inner_content(|writer| {
                    writer.create_element("empty").write_empty().map(|_| ())
                })?;
            Ok(())
        })?;

    assert_eq!(
        utf8(&buffer),
        r#"<outer attr1="value1" attr2="value2">
    <fruit quantity="0">apple</fruit>
    <fruit quantity="1">orange</fruit>
    <fruit quantity="2">banana</fruit>
    <inner>
        <empty/>
    </inner>
</outer>"#
    );
    Ok(())
}

mod in_attributes {
    use super::*;

    #[test]
    fn newline_first() -> TestResult {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b'_', 1);

        writer
            .create_element("element")
            .new_line()
            .with_attribute(("first", "1"))
            .with_attribute(("second", "2"))
            .new_line()
            .with_attribute(("third", "3"))
            .with_attribute(("fourth", "4"))
            .write_empty()?;

        assert_eq!(
            utf8(&buffer),
            "<element\
                \n_first=\"1\" second=\"2\"\
                \n_third=\"3\" fourth=\"4\"/>"
        );
        Ok(())
    }

    #[test]
    fn newline_inside() -> TestResult {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b'_', 1);

        writer
            .create_element("element")
            .with_attribute(("first", "1"))
            .with_attribute(("second", "2"))
            .new_line()
            .with_attribute(("third", "3"))
            .with_attribute(("fourth", "4"))
            .write_empty()?;

        assert_eq!(
            utf8(&buffer),
            "<element first=\"1\" second=\"2\"\
            \n         third=\"3\" fourth=\"4\"/>"
        );
        Ok(())
    }

    #[test]
    fn newline_last() -> TestResult {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b'_', 1);

        writer
            .create_element("element")
            .new_line()
            .with_attribute(("first", "1"))
            .with_attribute(("second", "2"))
            .new_line()
            .with_attribute(("third", "3"))
            .with_attribute(("fourth", "4"))
            .new_line()
            .write_empty()?;

        writer
            .create_element("element")
            .with_attribute(("first", "1"))
            .with_attribute(("second", "2"))
            .new_line()
            .with_attribute(("third", "3"))
            .with_attribute(("fourth", "4"))
            .new_line()
            .write_empty()?;

        assert_eq!(
            utf8(&buffer),
            "<element\
                \n_first=\"1\" second=\"2\"\
                \n_third=\"3\" fourth=\"4\"\
            \n/>\
            \n<element first=\"1\" second=\"2\"\
            \n         third=\"3\" fourth=\"4\"\
            \n/>"
        );
        Ok(())
    }

    #[test]
    fn newline_twice() -> TestResult {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b'_', 1);

        writer
            .create_element("element")
            .new_line()
            .new_line()
            .write_empty()?;

        writer
            .create_element("element")
            .with_attribute(("first", "1"))
            .new_line()
            .new_line()
            .with_attribute(("second", "2"))
            .write_empty()?;

        assert_eq!(
            utf8(&buffer),
            r#"<element

/>
<element first="1"

         second="2"/>"#
        );
        Ok(())
    }

    #[test]
    fn without_indent() -> TestResult {
        let mut buffer = Vec::new();
        let mut writer = Writer::new(&mut buffer);

        writer
            .create_element("element")
            .new_line()
            .new_line()
            .write_empty()?;

        writer
            .create_element("element")
            .with_attribute(("first", "1"))
            .new_line()
            .new_line()
            .with_attribute(("second", "2"))
            .write_empty()?;

        assert_eq!(
            utf8(&buffer),
            r#"<element/><element first="1" second="2"/>"#
        );
        Ok(())
    }

    /// The space-indent cache must grow past its initial capacity for very long
    /// element names.
    #[test]
    fn long_element_name() -> TestResult {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b't', 1);

        let name = "x".repeat(128);
        writer
            .create_element(name.as_str())
            .with_attribute(("first", "1"))
            .new_line()
            .with_attribute(("second", "2"))
            .write_empty()?;

        let expected = format!(
            "<{name} first=\"1\"\n{}second=\"2\"/>",
            " ".repeat(name.len() + 2),
        );
        assert_eq!(utf8(&buffer), expected);
        Ok(())
    }
}

mod in_attributes_multi {
    use super::*;

    #[test]
    fn newline_first() -> TestResult {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b'_', 1);

        writer
            .create_element("element")
            .new_line()
            .with_attributes([("first", "1"), ("second", "2")])
            .new_line()
            .with_attributes([("third", "3"), ("fourth", "4")])
            .write_empty()?;

        assert_eq!(
            utf8(&buffer),
            "<element\
                \n_first=\"1\" second=\"2\"\
                \n_third=\"3\" fourth=\"4\"/>"
        );
        Ok(())
    }

    #[test]
    fn newline_inside() -> TestResult {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b'_', 1);

        writer
            .create_element("element")
            .with_attributes([("first", "1"), ("second", "2")])
            .new_line()
            .with_attributes([("third", "3"), ("fourth", "4")])
            .write_empty()?;

        assert_eq!(
            utf8(&buffer),
            r#"<element first="1" second="2"
         third="3" fourth="4"/>"#
        );
        Ok(())
    }

    #[test]
    fn newline_last() -> TestResult {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b'_', 1);

        writer
            .create_element("element")
            .new_line()
            .with_attributes([("first", "1"), ("second", "2")])
            .new_line()
            .with_attributes([("third", "3"), ("fourth", "4")])
            .new_line()
            .write_empty()?;

        writer
            .create_element("element")
            .with_attributes([("first", "1"), ("second", "2")])
            .new_line()
            .with_attributes([("third", "3"), ("fourth", "4")])
            .new_line()
            .write_empty()?;

        assert_eq!(
            utf8(&buffer),
            "<element\
                \n_first=\"1\" second=\"2\"\
                \n_third=\"3\" fourth=\"4\"\
            \n/>\
            \n<element first=\"1\" second=\"2\"\
            \n         third=\"3\" fourth=\"4\"\
            \n/>"
        );
        Ok(())
    }
}
