//! Round-trip tests: writing the events produced by a [`Reader`] back out with a
//! [`Writer`] must reproduce an equivalent document.
//!
//! The per-event and document cases are ported from quick-xml 0.41
//! `tests/roundtrip.rs` (MIT/Apache-2.0, © the quick-xml authors), adapted to the
//! `oxixml_xml` public API and to the COOLJAPAN no-`unwrap()` policy (tests use
//! `?`/`expect`). The corpus tests are `oxixml`-specific: they replay the bundled
//! documents, and — when `OXIXML_XML_SUITES_DIR` points at a checkout of the W3C
//! XML suites — a bounded, deterministic sample of the suites, checking that the
//! event stream survives a read → write → re-read cycle unchanged.
//!
//! [`Reader`]: oxixml_xml::reader::Reader
//! [`Writer`]: oxixml_xml::writer::Writer

use std::path::{Path, PathBuf};

use oxixml_xml::errors::Error;
use oxixml_xml::events::Event::*;
use oxixml_xml::events::attributes::AttrError;
use oxixml_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use oxixml_xml::reader::{Config, Reader};
use oxixml_xml::writer::Writer;

type TestResult = Result<(), Error>;

/// Reads every event of `xml` into owned events, excluding the final `Eof`.
fn read_all(xml: &str, config: impl Fn(&mut Config)) -> Result<Vec<Event<'static>>, Error> {
    let mut reader = Reader::from_str(xml);
    config(reader.config_mut());
    let mut events = Vec::new();
    loop {
        match reader.read_event()? {
            Eof => break,
            e => events.push(e.into_owned()),
        }
    }
    Ok(events)
}

/// Writes a stream of events and returns the produced document.
fn write_all(events: &[Event<'_>], indent: Option<(u8, usize)>) -> String {
    let mut writer = match indent {
        Some((ch, size)) => Writer::new_with_indent(Vec::new(), ch, size),
        None => Writer::new(Vec::new()),
    };
    for event in events {
        writer.write_event(event.borrow()).expect("write event");
    }
    String::from_utf8(writer.into_inner()).expect("utf-8 output")
}

/// The reader configuration required for a byte-exact round trip.
///
/// The only default that rewrites the input is
/// [`trim_markup_names_in_closing_tags`](Config::trim_markup_names_in_closing_tags),
/// which drops the white space in `</a >`. Keeping that white space means the
/// raw end name no longer matches the start name literally, so
/// [`check_end_names`](Config::check_end_names) has to be relaxed as well.
fn verbatim(config: &mut Config) {
    config.trim_markup_names_in_closing_tags = false;
    config.check_end_names = false;
}

/// Reads `xml` and writes it straight back out, asserting the bytes are identical.
fn assert_verbatim(xml: &str, config: impl Fn(&mut Config) + Copy) -> TestResult {
    let events = read_all(xml, config)?;
    let written = write_all(&events, None);
    assert_eq!(written, xml, "written bytes differ from the input");
    // Re-reading must yield the very same event stream.
    assert_eq!(
        read_all(&written, config)?,
        events,
        "event stream changed after a round trip",
    );
    Ok(())
}

mod events {
    use super::*;

    /// `Start` and `End` are tested together, because reading a lone `End` event
    /// requires a special reader configuration.
    #[test]
    fn start_end() -> TestResult {
        assert_verbatim(
            r#"<source attr="val" attr2 = ' "-->&entity;<-- '></source>"#,
            |_| (),
        )
    }

    #[test]
    fn empty() -> TestResult {
        assert_verbatim(
            r#"<source attr="val" attr2 = ' "-->&entity;<-- '/>"#,
            |_| (),
        )
    }

    #[test]
    fn text() -> TestResult {
        assert_verbatim(
            "it is just arbitrary text &amp; some character reference",
            |_| (),
        )
    }

    #[test]
    fn cdata() -> TestResult {
        assert_verbatim("<![CDATA[text & no references]]>", |_| ())
    }

    #[test]
    fn pi() -> TestResult {
        assert_verbatim("<?!-- some strange processing instruction ?>", |_| ())
    }

    #[test]
    fn decl() -> TestResult {
        assert_verbatim("<?xml some strange XML declaration ?>", |_| ())
    }

    #[test]
    fn comment() -> TestResult {
        assert_verbatim("<!-- some comment with -- inside---->", |c| {
            c.check_comments = false;
        })
    }

    #[test]
    fn doctype() -> TestResult {
        assert_verbatim(r#"<!DOCTYPE greeting SYSTEM "hello.dtd">"#, |_| ())
    }

    #[test]
    fn general_ref() -> TestResult {
        assert_verbatim("<a>before&entity;after</a>", |_| ())
    }
}

/// The indentation of the last tag is mismatched intentionally: the writer must
/// not touch whitespace it did not produce itself.
const XML: &str = r#"
        <?xml version="1.0" encoding="UTF-8"?>
        <section ns:label="header">
            <section ns:label="empty element section" />
            <section ns:label="start/end section"></section>
            <section ns:label="with text">data &lt;escaped&gt;</section>
            </section>
    "#;

/// Writes events straight from the reader, without any processing.
#[test]
fn simple() -> TestResult {
    assert_verbatim(XML, |_| ())
}

/// Writes events straight from the reader, with text auto-trimming enabled.
#[test]
fn with_trim() -> TestResult {
    let input = include_str!("documents/flat.xml").trim();
    let events = read_all(input, |c| c.trim_text(true))?;
    assert_eq!(write_all(&events, None), input);
    Ok(())
}

/// The same, but writing borrowed events (`&Event`) instead of owned ones.
#[test]
fn with_trim_ref() -> TestResult {
    let input = include_str!("documents/flat.xml").trim();
    let mut reader = Reader::from_str(input);
    reader.config_mut().trim_text(true);
    let mut writer = Writer::new(Vec::new());
    loop {
        match reader.read_event()? {
            Eof => break,
            e => writer.write_event(e.borrow()).expect("write event"),
        }
    }
    assert_eq!(
        String::from_utf8(writer.into_inner()).expect("utf-8 output"),
        input,
    );
    Ok(())
}

/// Re-indenting a document with the same settings it was written with must
/// reproduce it byte for byte.
#[test]
fn with_indent() -> TestResult {
    let input = include_str!("documents/indent.xml");
    let events = read_all(input, |c| c.trim_text(true))?;
    assert_eq!(write_all(&events, Some((b' ', 4))), input);
    Ok(())
}

/// The same, for a document that contains CDATA sections (which must stay on the
/// line of their enclosing element).
#[test]
fn with_indent_cdata() -> TestResult {
    let input = include_str!("documents/indent_cdata.xml");
    let events = read_all(input, |c| c.trim_text(true))?;
    assert_eq!(write_all(&events, Some((b' ', 4))), input);
    Ok(())
}

/// Unescaping and re-escaping the content of every `Text` event is a no-op for a
/// document that uses the canonical escapes.
#[test]
fn reescape_text() -> TestResult {
    let mut reader = Reader::from_str(XML);
    let mut writer = Writer::new(Vec::new());
    loop {
        match reader.read_event()? {
            Eof => break,
            Text(e) => {
                let text = e.xml10_content()?;
                writer
                    .write_event(Text(BytesText::new(&text)))
                    .expect("write text");
            }
            e => writer.write_event(e).expect("write event"),
        }
    }
    assert_eq!(
        String::from_utf8(writer.into_inner()).expect("utf-8 output"),
        XML,
    );
    Ok(())
}

/// Rewrites some of the events while streaming them through the writer.
#[test]
fn partial_rewrite() -> TestResult {
    let input = r#"<source attr="val"></source>"#;
    let expected = r#"<copy attr="val" a="b" c="d" x="y&quot;z"></copy>"#;

    let mut reader = Reader::from_str(input);
    let mut writer = Writer::new(Vec::new());
    loop {
        let event = match reader.read_event()? {
            Eof => break,
            Start(element) => {
                let mut attrs = element
                    .attributes()
                    .collect::<Result<Vec<_>, AttrError>>()
                    .map_err(Error::InvalidAttr)?;
                attrs.extend_from_slice(&[("a", "b").into(), ("c", "d").into()]);
                let mut element = BytesStart::new("copy");
                element.extend_attributes(attrs);
                element.push_attribute(("x", "y\"z"));
                Start(element)
            }
            End(_) => End(BytesEnd::new("copy")),
            e => e,
        };
        writer.write_event(event).expect("write event");
    }
    assert_eq!(
        String::from_utf8(writer.into_inner()).expect("utf-8 output"),
        expected,
    );
    Ok(())
}

/// White space between the name and the `>` of a closing tag is dropped by the
/// default reader configuration, so a verbatim round trip has to turn
/// [`trim_markup_names_in_closing_tags`](Config::trim_markup_names_in_closing_tags)
/// off.
#[test]
fn closing_tag_white_space() -> TestResult {
    let xml = "<a>text</a\n  >";

    let trimmed = read_all(xml, |_| ())?;
    assert_eq!(write_all(&trimmed, None), "<a>text</a>");

    let kept = read_all(xml, verbatim)?;
    assert_eq!(write_all(&kept, None), xml);
    Ok(())
}

/// A leading UTF-8 BOM is consumed by the reader and is re-emitted only through
/// an explicit [`write_bom`](Writer::write_bom) call.
#[test]
fn bom_round_trip() -> TestResult {
    let xml = "\u{FEFF}<?xml version=\"1.0\"?><root>text</root>";
    let events = read_all(xml, |_| ())?;

    // Without `write_bom`, the marker is gone.
    assert_eq!(write_all(&events, None), &xml[3..]);

    // With it, the document is reproduced byte for byte.
    let mut writer = Writer::new(Vec::new());
    writer.write_bom().expect("write bom");
    for event in &events {
        writer.write_event(event.borrow()).expect("write event");
    }
    assert_eq!(
        String::from_utf8(writer.into_inner()).expect("utf-8 output"),
        xml,
    );
    Ok(())
}

/// A hand-written corpus that covers every event kind and both quoting styles.
#[test]
fn corpus_of_snippets() -> TestResult {
    const SNIPPETS: &[&str] = &[
        r#"<root>text</root>"#,
        r#"<a k1="v1" k2='v2'><b/><c>x</c></a>"#,
        r#"<x:a xmlns:x="urn:x"><x:b>t</x:b></x:a>"#,
        r#"<a><!--a comment--></a>"#,
        r#"<a><![CDATA[ raw <b> & ]]></a>"#,
        r#"<a>one &amp; two &lt; three</a>"#,
        r#"<?xml version="1.0" encoding="UTF-8"?><root/>"#,
        r#"<a><?target instruction?></a>"#,
        "<a>  \n  <b/>\n</a>",
        r#"<a self="closing"/>"#,
        r#"<a b = "spaced" c ='out' />"#,
        r#"<!DOCTYPE a [<!ENTITY e "v">]><a>&e;</a>"#,
        "<a\n b=\"multi\n line\"/>",
        "<a>\u{1F600} unicode \u{3042}\u{3044}</a>",
    ];

    for snippet in SNIPPETS {
        assert_verbatim(snippet, |_| ())?;
    }
    Ok(())
}

/// Every bundled document must survive a read → write → re-read cycle with the
/// event stream unchanged.
#[test]
fn corpus_of_documents() -> TestResult {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/documents");
    let mut documents = std::fs::read_dir(&dir)
        .expect("the bundled document corpus is readable")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "xml"))
        .collect::<Vec<_>>();
    documents.sort();
    assert!(
        !documents.is_empty(),
        "the document corpus must not be empty"
    );

    for path in documents {
        let xml = std::fs::read_to_string(&path).expect("document is valid UTF-8");
        assert_verbatim(&xml, |_| ())
            .unwrap_or_else(|e| panic!("{} did not round-trip: {e}", path.display()));
    }
    Ok(())
}

/// The maximum number of suite documents replayed by [`corpus_of_w3c_suites`],
/// keeping the test bounded regardless of the size of the checkout.
const SUITE_SAMPLE_LIMIT: usize = 1500;

/// Collects up to `limit` `*.xml` files under `root`, walking directories in a
/// deterministic (sorted) order.
fn collect_xml_files(root: &Path, limit: usize) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut queue = vec![root.to_path_buf()];
    while let Some(dir) = queue.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut children = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .collect::<Vec<_>>();
        children.sort();
        for path in children {
            if path.is_dir() {
                queue.push(path);
            } else if path.extension().is_some_and(|ext| ext == "xml") {
                found.push(path);
                if found.len() >= limit {
                    return found;
                }
            }
        }
    }
    found.sort();
    found
}

/// Replays a bounded sample of the W3C XML suites through the reader and writer.
///
/// The suites deliberately contain malformed documents and documents in
/// encodings that the default (UTF-8 only) build cannot decode; those are
/// skipped, since only documents that parse cleanly can be round-tripped. The
/// test is skipped entirely when `OXIXML_XML_SUITES_DIR` is not set.
#[test]
fn corpus_of_w3c_suites() {
    let Ok(root) = std::env::var("OXIXML_XML_SUITES_DIR") else {
        // Suite not configured — nothing to do.
        return;
    };
    let root = PathBuf::from(root);
    if !root.is_dir() {
        return;
    }

    let mut checked = 0usize;
    for path in collect_xml_files(&root, SUITE_SAMPLE_LIMIT) {
        let Ok(xml) = std::fs::read_to_string(&path) else {
            continue; // Not UTF-8 — out of scope for the default build.
        };
        // The reader consumes a leading BOM, and the writer only re-emits one on
        // an explicit `write_bom()` call, so compare against the stripped text.
        let expected = xml.strip_prefix('\u{FEFF}').unwrap_or(&xml);
        // Only well-formed-enough documents can be replayed verbatim.
        let Ok(events) = read_all(expected, verbatim) else {
            continue;
        };
        let written = write_all(&events, None);
        assert_eq!(written, expected, "{} was not reproduced", path.display());
        let reparsed = read_all(&written, verbatim)
            .unwrap_or_else(|e| panic!("{} did not re-parse: {e}", path.display()));
        assert_eq!(
            reparsed,
            events,
            "{} changed its event stream",
            path.display(),
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "OXIXML_XML_SUITES_DIR is set but no document could be replayed",
    );
}
