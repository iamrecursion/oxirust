//! Integration tests for the pull [`Reader`].
//!
//! Some cases are ported from quick-xml 0.41 `tests/reader.rs` and
//! `tests/reader-errors.rs` (MIT/Apache-2.0, © the quick-xml authors), adapted to
//! the `oxixml_xml` public API and to the COOLJAPAN no-`unwrap()` policy (tests
//! use `?`/`expect`). The remainder are `oxixml`-specific: deep nesting, huge
//! attributes, chunk-boundary reads through a one-byte `BufRead`, exhaustive
//! `Config` toggles and byte-position reporting.

use std::io::{self, BufRead, Read};

use oxixml_xml::errors::{Error, IllFormedError, SyntaxError};
use oxixml_xml::events::{BytesEnd, BytesRef, BytesStart, Event};
use oxixml_xml::reader::Reader;

/// A [`BufRead`] that hands out at most one byte at a time from `fill_buf`, to
/// force every markup construct across a chunk boundary. Exercises the buffered
/// `XmlSource` state machine's split-sequence handling.
struct OneByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> OneByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl Read for OneByteReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let n = available.len().min(out.len());
        out[..n].copy_from_slice(&available[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl BufRead for OneByteReader<'_> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        let end = (self.pos + 1).min(self.data.len());
        Ok(&self.data[self.pos..end])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = (self.pos + amt).min(self.data.len());
    }
}

/// Collects all events from a borrowed (`&str`) reader into owned events.
fn collect_borrowed(xml: &str) -> Result<Vec<Event<'static>>, Error> {
    let mut reader = Reader::from_str(xml);
    let mut events = Vec::new();
    loop {
        let event = reader.read_event()?;
        let is_eof = matches!(event, Event::Eof);
        events.push(event.into_owned());
        if is_eof {
            break;
        }
    }
    Ok(events)
}

/// Collects all events from a buffered reader that yields one byte at a time.
fn collect_one_byte(xml: &str) -> Result<Vec<Event<'static>>, Error> {
    let mut reader = Reader::from_reader(OneByteReader::new(xml.as_bytes()));
    let mut buf = Vec::new();
    let mut events = Vec::new();
    loop {
        let event = reader.read_event_into(&mut buf)?;
        let is_eof = matches!(event, Event::Eof);
        events.push(event.into_owned());
        if is_eof {
            break;
        }
        buf.clear();
    }
    Ok(events)
}

// -----------------------------------------------------------------------------
// Basic event shapes (ported from quick-xml tests/reader.rs)
// -----------------------------------------------------------------------------

#[test]
fn start_end() -> Result<(), Error> {
    let mut r = Reader::from_str("<a></a>");
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("a")));
    assert_eq!(r.read_event()?, Event::Eof);
    Ok(())
}

#[test]
fn start_end_with_trailing_ws_in_close() -> Result<(), Error> {
    // `trim_markup_names_in_closing_tags` defaults to true.
    let mut r = Reader::from_str("<a></a >");
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("a")));
    Ok(())
}

#[test]
fn start_end_attr() -> Result<(), Error> {
    let mut r = Reader::from_str("<a b=\"test\"></a>");
    assert_eq!(
        r.read_event()?,
        Event::Start(BytesStart::from_content("a b=\"test\"", 1))
    );
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("a")));
    Ok(())
}

#[test]
fn empty_element() -> Result<(), Error> {
    let mut r = Reader::from_str("<a b=\"test\" />");
    assert_eq!(
        r.read_event()?,
        Event::Empty(BytesStart::from_content("a b=\"test\" ", 1))
    );
    assert_eq!(r.read_event()?, Event::Eof);
    Ok(())
}

#[test]
fn escaped_content_yields_general_refs() -> Result<(), Error> {
    // `&lt;test&gt;` becomes: ref, text, ref.
    let mut r = Reader::from_str("<a>&lt;test&gt;</a>");
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    assert_eq!(r.read_event()?, Event::GeneralRef(BytesRef::new("lt")));
    match r.read_event()? {
        Event::Text(e) => assert_eq!(&*e, b"test"),
        other => panic!("expected text, got {other:?}"),
    }
    assert_eq!(r.read_event()?, Event::GeneralRef(BytesRef::new("gt")));
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("a")));
    assert_eq!(r.read_event()?, Event::Eof);
    Ok(())
}

// -----------------------------------------------------------------------------
// EOF forever
// -----------------------------------------------------------------------------

#[test]
fn eof_is_reported_forever() -> Result<(), Error> {
    let mut r = Reader::from_str("<a/>");
    assert_eq!(r.read_event()?, Event::Empty(BytesStart::new("a")));
    for _ in 0..5 {
        assert_eq!(r.read_event()?, Event::Eof);
    }
    Ok(())
}

#[test]
fn eof_forever_after_error() -> Result<(), Error> {
    // A syntax error moves the reader to the terminal `Done` state; further
    // reads yield `Eof`.
    let mut r = Reader::from_str("<a");
    assert!(matches!(
        r.read_event(),
        Err(Error::Syntax(SyntaxError::UnclosedTag))
    ));
    assert_eq!(r.read_event()?, Event::Eof);
    assert_eq!(r.read_event()?, Event::Eof);
    Ok(())
}

// -----------------------------------------------------------------------------
// No CR/LF normalization inside the reader
// -----------------------------------------------------------------------------

#[test]
fn text_crlf_is_not_normalized() -> Result<(), Error> {
    let mut r = Reader::from_str("<a>x\r\ny\r z</a>");
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    match r.read_event()? {
        // The reader returns raw bytes; end-of-line normalization is a decode
        // concern, not a tokenization concern.
        Event::Text(e) => assert_eq!(&*e, b"x\r\ny\r z"),
        other => panic!("expected text, got {other:?}"),
    }
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("a")));
    Ok(())
}

// -----------------------------------------------------------------------------
// XML declaration accessors
// -----------------------------------------------------------------------------

#[test]
fn declaration_accessors() -> Result<(), Error> {
    let mut r = Reader::from_str(r#"<?xml version="1.1" encoding="UTF-8" standalone="yes"?><a/>"#);
    match r.read_event()? {
        Event::Decl(decl) => {
            assert_eq!(&*decl.version()?, b"1.1");
            let encoding = decl.encoding().expect("encoding present")?;
            assert_eq!(&*encoding, b"UTF-8");
            let standalone = decl.standalone().expect("standalone present")?;
            assert_eq!(&*standalone, b"yes");
        }
        other => panic!("expected declaration, got {other:?}"),
    }
    assert_eq!(r.read_event()?, Event::Empty(BytesStart::new("a")));
    Ok(())
}

// -----------------------------------------------------------------------------
// DOCTYPE, including a bracketed internal subset
// -----------------------------------------------------------------------------

#[test]
fn doctype_simple() -> Result<(), Error> {
    let mut r = Reader::from_str("<!DOCTYPE greeting>");
    match r.read_event()? {
        Event::DocType(e) => assert_eq!(&*e, b"greeting"),
        other => panic!("expected doctype, got {other:?}"),
    }
    Ok(())
}

#[test]
fn doctype_internal_subset() -> Result<(), Error> {
    // The `>` in `<!ELEMENT ...>` inside the internal subset must not end the
    // DOCTYPE early.
    let xml = "<!DOCTYPE r [ <!ELEMENT r (#PCDATA)> <!-- c > --> ]><r/>";
    let mut r = Reader::from_str(xml);
    match r.read_event()? {
        Event::DocType(e) => {
            assert_eq!(&*e, b"r [ <!ELEMENT r (#PCDATA)> <!-- c > --> ]")
        }
        other => panic!("expected doctype, got {other:?}"),
    }
    assert_eq!(r.read_event()?, Event::Empty(BytesStart::new("r")));
    Ok(())
}

#[test]
fn doctype_case_insensitive_keyword() -> Result<(), Error> {
    // The `DOCTYPE` keyword is matched case-insensitively.
    let mut r = Reader::from_str("<!doctype x>");
    match r.read_event()? {
        Event::DocType(e) => assert_eq!(&*e, b"x"),
        other => panic!("expected doctype, got {other:?}"),
    }
    Ok(())
}

#[test]
fn doctype_internal_subset_across_one_byte_chunks() -> Result<(), Error> {
    let xml = "<!DOCTYPE r [ <!ENTITY e \"a > b\"> <!ATTLIST r a CDATA #IMPLIED> ]><r/>";
    let borrowed = collect_borrowed(xml)?;
    let chunked = collect_one_byte(xml)?;
    assert_eq!(borrowed, chunked);
    match &borrowed[0] {
        Event::DocType(_) => {}
        other => panic!("expected doctype, got {other:?}"),
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Ill-formedness: mismatched / unmatched / unclosed
// -----------------------------------------------------------------------------

#[test]
fn mismatched_end_tag_is_detected() {
    let mut r = Reader::from_str("<a></b>");
    assert!(matches!(r.read_event(), Ok(Event::Start(_))));
    match r.read_event() {
        Err(Error::IllFormed(IllFormedError::MismatchedEndTag { expected, found })) => {
            assert_eq!(expected, "a");
            assert_eq!(found, "b");
        }
        other => panic!("expected MismatchedEndTag, got {other:?}"),
    }
    // Error is reported at the `<` of the `</b>` end tag.
    assert_eq!(r.error_position(), 3);
}

#[test]
fn mismatched_end_tag_ignored_when_disabled() -> Result<(), Error> {
    let mut r = Reader::from_str("<a></b>");
    r.config_mut().check_end_names = false;
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("b")));
    Ok(())
}

#[test]
fn unmatched_end_tag_is_detected() {
    let mut r = Reader::from_str("</a>");
    match r.read_event() {
        Err(Error::IllFormed(IllFormedError::UnmatchedEndTag(name))) => assert_eq!(name, "a"),
        other => panic!("expected UnmatchedEndTag, got {other:?}"),
    }
    assert_eq!(r.error_position(), 0);
}

#[test]
fn unmatched_end_tag_allowed_when_enabled() -> Result<(), Error> {
    let mut r = Reader::from_str("<a></a></b>");
    r.config_mut().allow_unmatched_ends = true;
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("a")));
    // The unmatched `</b>` is emitted unchanged.
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("b")));
    assert_eq!(r.read_event()?, Event::Eof);
    Ok(())
}

#[test]
fn unclosed_tag_reports_at_open_angle() {
    let mut r = Reader::from_str("<a");
    assert!(matches!(
        r.read_event(),
        Err(Error::Syntax(SyntaxError::UnclosedTag))
    ));
    assert_eq!(r.error_position(), 0);
    assert_eq!(r.buffer_position(), 2);
}

#[test]
fn missing_end_tag_from_read_to_end() {
    let mut r = Reader::from_str("<a><b>");
    assert!(matches!(r.read_event(), Ok(Event::Start(_))));
    let start = BytesStart::new("a");
    match r.read_to_end(start.name()) {
        Err(Error::IllFormed(IllFormedError::MissingEndTag(name))) => assert_eq!(name, "a"),
        other => panic!("expected MissingEndTag, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// Comment double-hyphen validation
// -----------------------------------------------------------------------------

#[test]
fn comment_double_hyphen_allowed_by_default() -> Result<(), Error> {
    let mut r = Reader::from_str("<!-- a -- b -->");
    match r.read_event()? {
        Event::Comment(e) => assert_eq!(&*e, b" a -- b "),
        other => panic!("expected comment, got {other:?}"),
    }
    Ok(())
}

#[test]
fn comment_double_hyphen_rejected_when_checked() {
    let mut r = Reader::from_str("<!-- a -- b -->");
    r.config_mut().check_comments = true;
    assert!(matches!(
        r.read_event(),
        Err(Error::IllFormed(IllFormedError::DoubleHyphenInComment))
    ));
}

// -----------------------------------------------------------------------------
// Dangling ampersand
// -----------------------------------------------------------------------------

#[test]
fn dangling_amp_rejected_by_default() {
    let mut r = Reader::from_str("<a>x & y</a>");
    assert!(matches!(r.read_event(), Ok(Event::Start(_))));
    // Text up to `&`
    assert!(matches!(r.read_event(), Ok(Event::Text(_))));
    assert!(matches!(
        r.read_event(),
        Err(Error::IllFormed(IllFormedError::UnclosedReference))
    ));
}

#[test]
fn dangling_amp_allowed_when_enabled() -> Result<(), Error> {
    let mut r = Reader::from_str("text with & &amp; & alone");
    r.config_mut().allow_dangling_amp = true;
    match r.read_event()? {
        Event::Text(e) => assert_eq!(&*e, b"text with "),
        other => panic!("expected text, got {other:?}"),
    }
    match r.read_event()? {
        Event::Text(e) => assert_eq!(&*e, b"& "),
        other => panic!("expected text, got {other:?}"),
    }
    assert_eq!(r.read_event()?, Event::GeneralRef(BytesRef::new("amp")));
    match r.read_event()? {
        Event::Text(e) => assert_eq!(&*e, b" "),
        other => panic!("expected text, got {other:?}"),
    }
    match r.read_event()? {
        Event::Text(e) => assert_eq!(&*e, b"& alone"),
        other => panic!("expected text, got {other:?}"),
    }
    assert_eq!(r.read_event()?, Event::Eof);
    Ok(())
}

// -----------------------------------------------------------------------------
// expand_empty_elements
// -----------------------------------------------------------------------------

#[test]
fn expand_empty_elements_splits_events() -> Result<(), Error> {
    let mut r = Reader::from_str("<a/>");
    r.config_mut().expand_empty_elements = true;
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("a")));
    assert_eq!(r.read_event()?, Event::Eof);
    Ok(())
}

// -----------------------------------------------------------------------------
// trim_text
// -----------------------------------------------------------------------------

#[test]
fn trim_text_drops_whitespace_only_events() -> Result<(), Error> {
    let mut r = Reader::from_str("<a>  <b>  hello  </b>  </a>");
    r.config_mut().trim_text(true);
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("b")));
    match r.read_event()? {
        Event::Text(e) => assert_eq!(&*e, b"hello"),
        other => panic!("expected trimmed text, got {other:?}"),
    }
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("b")));
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("a")));
    assert_eq!(r.read_event()?, Event::Eof);
    Ok(())
}

#[test]
fn trim_text_end_only_keeps_leading_ws() -> Result<(), Error> {
    let mut r = Reader::from_str("<a>  hi  </a>");
    r.config_mut().trim_text_end = true;
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    match r.read_event()? {
        Event::Text(e) => assert_eq!(&*e, b"  hi"),
        other => panic!("expected text, got {other:?}"),
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Byte position reporting
// -----------------------------------------------------------------------------

#[test]
fn buffer_position_tracks_event_ends() -> Result<(), Error> {
    let mut r = Reader::from_str("<tag>abc</tag>");
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("tag")));
    assert_eq!(r.buffer_position(), 5);
    assert!(matches!(r.read_event()?, Event::Text(_)));
    assert_eq!(r.buffer_position(), 8);
    assert_eq!(r.read_event()?, Event::End(BytesEnd::new("tag")));
    assert_eq!(r.buffer_position(), 14);
    Ok(())
}

#[test]
fn positions_are_bytes_not_chars() -> Result<(), Error> {
    // A multi-byte character contributes its byte length to the position.
    let mut r = Reader::from_str("<a>é</a>");
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    assert!(matches!(r.read_event()?, Event::Text(_)));
    // `é` is two bytes in UTF-8, so text ends at 3 + 2 = 5.
    assert_eq!(r.buffer_position(), 5);
    Ok(())
}

// -----------------------------------------------------------------------------
// read_to_end / read_text
// -----------------------------------------------------------------------------

#[test]
fn read_to_end_skips_nested_same_name() -> Result<(), Error> {
    let mut r = Reader::from_str("<a><a></a><a/></a><b/>");
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    let end = BytesStart::new("a").to_end().into_owned();
    r.read_to_end(end.name())?;
    assert_eq!(r.read_event()?, Event::Empty(BytesStart::new("b")));
    assert_eq!(r.read_event()?, Event::Eof);
    Ok(())
}

#[test]
fn read_text_returns_inner_markup_verbatim() -> Result<(), Error> {
    let mut r = Reader::from_str("<a>x<b/>y</a>");
    assert_eq!(r.read_event()?, Event::Start(BytesStart::new("a")));
    let end = BytesStart::new("a").to_end().into_owned();
    let text = r.read_text(end.name())?;
    assert_eq!(&*text, b"x<b/>y");
    Ok(())
}

#[test]
fn read_to_end_into_buffered() -> Result<(), Error> {
    let mut r = Reader::from_reader("<a><inner/></a><tail/>".as_bytes());
    let mut buf = Vec::new();
    assert_eq!(
        r.read_event_into(&mut buf)?,
        Event::Start(BytesStart::new("a"))
    );
    buf.clear();
    let end = BytesStart::new("a").to_end().into_owned();
    r.read_to_end_into(end.name(), &mut buf)?;
    buf.clear();
    assert_eq!(
        r.read_event_into(&mut buf)?,
        Event::Empty(BytesStart::new("tail"))
    );
    Ok(())
}

// -----------------------------------------------------------------------------
// Chunk-boundary equivalence (one byte at a time == borrowed)
// -----------------------------------------------------------------------------

#[test]
fn one_byte_chunks_match_borrowed() -> Result<(), Error> {
    let docs = [
        "<?xml version=\"1.0\"?><root a=\"1\" b='2'><child>text</child><!--c--></root>",
        "<a>&amp;&lt;&#48;</a>",
        "<![CDATA[ raw ]] ]]> data ]]><x/>",
        "<!-- comment with -> inside --><y/>",
        "<?target data with ? and > ?><z/>",
        "<a>line1\r\nline2</a>",
    ];
    for doc in docs {
        let borrowed = collect_borrowed(doc)?;
        let chunked = collect_one_byte(doc)?;
        assert_eq!(borrowed, chunked, "mismatch for doc: {doc:?}");
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Stress: deep nesting and huge attributes
// -----------------------------------------------------------------------------

#[test]
fn deep_nesting() -> Result<(), Error> {
    const DEPTH: usize = 5000;
    let mut xml = String::new();
    for _ in 0..DEPTH {
        xml.push_str("<a>");
    }
    for _ in 0..DEPTH {
        xml.push_str("</a>");
    }

    let mut r = Reader::from_str(&xml);
    let mut starts = 0;
    let mut ends = 0;
    loop {
        match r.read_event()? {
            Event::Start(_) => starts += 1,
            Event::End(_) => ends += 1,
            Event::Eof => break,
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert_eq!(starts, DEPTH);
    assert_eq!(ends, DEPTH);
    Ok(())
}

#[test]
fn huge_attribute_value() -> Result<(), Error> {
    let value = "x".repeat(1_000_000);
    let xml = format!("<a data=\"{value}\"/>");
    let mut r = Reader::from_str(&xml);
    match r.read_event()? {
        Event::Empty(e) => {
            let attr = e
                .attributes()
                .next()
                .expect("one attribute")
                .map_err(Error::from)?;
            assert_eq!(attr.value.len(), value.len());
        }
        other => panic!("expected empty element, got {other:?}"),
    }
    assert_eq!(r.read_event()?, Event::Eof);
    Ok(())
}

#[test]
fn huge_attribute_value_one_byte_chunks() -> Result<(), Error> {
    let value = "y".repeat(100_000);
    let xml = format!("<a data=\"{value}\"/>");
    let mut r = Reader::from_reader(OneByteReader::new(xml.as_bytes()));
    let mut buf = Vec::new();
    match r.read_event_into(&mut buf)? {
        Event::Empty(e) => assert_eq!(e.name().as_ref(), b"a"),
        other => panic!("expected empty element, got {other:?}"),
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// from_file
// -----------------------------------------------------------------------------

#[test]
fn from_file_reads_document() -> Result<(), Error> {
    use std::io::Write;

    let mut path = std::env::temp_dir();
    path.push(format!(
        "oxixml_reader_from_file_{}.xml",
        std::process::id()
    ));
    {
        let mut file = std::fs::File::create(&path)?;
        file.write_all(b"<root><child/></root>")?;
    }

    let mut r = Reader::from_file(&path)?;
    let mut buf = Vec::new();
    assert_eq!(
        r.read_event_into(&mut buf)?,
        Event::Start(BytesStart::new("root"))
    );
    buf.clear();
    assert_eq!(
        r.read_event_into(&mut buf)?,
        Event::Empty(BytesStart::new("child"))
    );
    buf.clear();
    assert_eq!(
        r.read_event_into(&mut buf)?,
        Event::End(BytesEnd::new("root"))
    );

    std::fs::remove_file(&path)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Cloning preserves independent state
// -----------------------------------------------------------------------------

#[test]
fn clone_has_independent_state() -> Result<(), Error> {
    let mut reader = Reader::from_str("<tag>text</tag>");
    assert_eq!(reader.read_event()?, Event::Start(BytesStart::new("tag")));

    let mut clone = reader.clone();
    assert!(matches!(reader.read_event()?, Event::Text(_)));
    // The clone is still positioned right after the start tag.
    assert!(matches!(clone.read_event()?, Event::Text(_)));
    assert_eq!(clone.read_event()?, Event::End(BytesEnd::new("tag")));
    assert_eq!(reader.read_event()?, Event::End(BytesEnd::new("tag")));
    Ok(())
}

// -----------------------------------------------------------------------------
// Trimming never produces an empty text event
// -----------------------------------------------------------------------------

/// Collects the debug representation of every event of `xml`, read with the
/// given trimming configuration, from both a slice and a buffered source.
fn events_with_trimming(xml: &str, trim_start: bool, trim_end: bool) -> Vec<String> {
    let mut from_slice = Reader::from_str(xml);
    from_slice.config_mut().trim_text_start = trim_start;
    from_slice.config_mut().trim_text_end = trim_end;
    let mut slice_events = Vec::new();
    loop {
        match from_slice.read_event().expect("event") {
            Event::Eof => break,
            other => slice_events.push(format!("{other:?}")),
        }
    }

    let mut from_reader = Reader::from_reader(std::io::BufReader::new(xml.as_bytes()));
    from_reader.config_mut().trim_text_start = trim_start;
    from_reader.config_mut().trim_text_end = trim_end;
    let mut buf = Vec::new();
    let mut buffered_events = Vec::new();
    loop {
        buf.clear();
        match from_reader.read_event_into(&mut buf).expect("event") {
            Event::Eof => break,
            other => buffered_events.push(format!("{other:?}")),
        }
    }

    assert_eq!(
        slice_events, buffered_events,
        "the slice and buffered readers must agree"
    );
    slice_events
}

/// With `trim_text_end` alone, an all-white-space run between two pieces of
/// markup (or before a reference) is trimmed to nothing, and no empty
/// `Text` event is emitted for it — as `Config::trim_text_end` documents.
#[test]
fn trimming_never_emits_an_empty_text_event() {
    for (trim_start, trim_end) in [(false, true), (true, false), (true, true)] {
        for event in events_with_trimming("<a>  <b/>  &amp;  </a>", trim_start, trim_end) {
            assert!(
                !event.contains("Text(BytesText { content: Borrowed(\"\") })"),
                "empty text event with trim_text_start={trim_start} trim_text_end={trim_end}: \
                 {event}",
            );
        }
    }

    // Without trimming every white-space run is still reported verbatim.
    assert_eq!(
        events_with_trimming("<a>  <b/></a>", false, false).len(),
        4,
        "untrimmed reading must keep the white-space text event"
    );

    // Text that survives trimming is unaffected.
    let events = events_with_trimming("<a>  text  <b/></a>", false, true);
    assert!(events[1].contains("Borrowed(\"  text\")"), "{events:?}");
}

// -----------------------------------------------------------------------------
// End of input is a terminal state, even with the well-formedness checks on
// -----------------------------------------------------------------------------

/// Reads at most `cap` events from `text` with the conformance-grade checks enabled, returning
/// how many reads it took to reach [`Event::Eof`] — `None` when the reader never got there.
fn reads_until_eof(text: &str, cap: usize) -> Option<usize> {
    let mut reader = Reader::from_str(text);
    reader.config_mut().check_well_formedness = true;
    for read in 1..=cap {
        if let Ok(Event::Eof) = reader.read_event() {
            return Some(read);
        }
    }
    None
}

/// `Error::IllFormed` is recoverable — the reader deliberately leaves its state alone so that a
/// caller may keep reading — so every end-of-input constraint has to be reported *once* and then
/// hold. Reporting it on every read made `read_event` a non-terminating loop for any caller that
/// kept going (and an unbounded allocator for one that collected a diagnostic per error).
#[test]
fn an_unclosed_document_still_reaches_eof() {
    for text in ["<a>", "<a><b>", "<a><b></a>", "<a><b><c><d>", ""] {
        let reads = reads_until_eof(text, 1_000);
        assert!(
            reads.is_some_and(|reads| reads <= 16),
            "{text:?} must reach Eof in a bounded number of reads, took {reads:?}"
        );
    }
}

/// Once `Eof` has been reported, it stays reported: no further read may resurrect the
/// end-of-input error.
#[test]
fn eof_stays_eof_after_the_unclosed_element_is_reported() {
    let mut reader = Reader::from_str("<a><b>");
    reader.config_mut().check_well_formedness = true;
    assert!(matches!(reader.read_event(), Ok(Event::Start(_))));
    assert!(matches!(reader.read_event(), Ok(Event::Start(_))));
    match reader.read_event() {
        Err(Error::IllFormed(IllFormedError::MissingEndTag(name))) => assert_eq!(
            name, "b",
            "the innermost open element is the one whose end tag is missing here"
        ),
        other => panic!("expected MissingEndTag, got {other:?}"),
    }
    for _ in 0..8 {
        assert!(matches!(reader.read_event(), Ok(Event::Eof)));
    }
}

/// A document with no root element is likewise reported once and then ends.
#[test]
fn a_document_without_a_root_element_is_reported_once() {
    let mut reader = Reader::from_str("<!-- just a comment -->");
    reader.config_mut().check_well_formedness = true;
    assert!(matches!(reader.read_event(), Ok(Event::Comment(_))));
    assert!(
        reader.read_event().is_err(),
        "a document must have a root element"
    );
    for _ in 0..4 {
        assert!(matches!(reader.read_event(), Ok(Event::Eof)));
    }
}
