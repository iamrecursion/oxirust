//! Integration tests for the skipping methods of [`NsReader`]:
//! [`read_to_end`], [`read_to_end_into`], [`read_text`] and [`read_text_into`].
//!
//! Ported from the `read_to_end`, `read_to_end_into`, `read_text` and
//! `read_text_into` modules of quick-xml 0.41 `tests/reader-namespaces.rs`
//! (MIT/Apache-2.0, © the quick-xml authors), adapted to the `oxixml_xml` public
//! API and to the COOLJAPAN no-`unwrap()` policy (tests use `?`/`expect`).
//!
//! Each scenario feeds the *same* document skeleton
//!
//! ```text
//! <root xmlns='namespace'>{fragment}<root/><root></root></root><element/>
//! ```
//!
//! with a different `{fragment}` in front of the nested `<root>` elements, so
//! that every event kind is exercised as the "current" event when the skipping
//! method is called. The nested elements verify that the methods match the
//! *literally* same name rather than the first `</root>` they encounter.
//!
//! [`NsReader`]: oxixml_xml::reader::NsReader
//! [`read_to_end`]: oxixml_xml::reader::NsReader::read_to_end
//! [`read_to_end_into`]: oxixml_xml::reader::NsReader::read_to_end_into
//! [`read_text`]: oxixml_xml::reader::NsReader::read_text
//! [`read_text_into`]: oxixml_xml::reader::NsReader::read_text_into

use std::ops::Range;

use oxixml_xml::errors::Error;
use oxixml_xml::events::Event::*;
use oxixml_xml::events::{
    BytesCData, BytesDecl, BytesEnd, BytesPI, BytesRef, BytesStart, BytesText, Event,
};
use oxixml_xml::name::ResolveResult::*;
use oxixml_xml::name::{Namespace, QName, ResolveResult};
use oxixml_xml::reader::NsReader;

type TestResult = Result<(), Error>;

/// The document prefix that opens the namespace scope.
const PREFIX: &str = "<root xmlns='namespace'>";
/// The nested elements skipped over by the method under test.
const SKIPPED: &str = "<root/><root></root>";

/// One row of the skipping matrix.
struct Scenario {
    /// Human readable name, used in assertion messages.
    name: &'static str,
    /// Markup inserted between the opening `<root>` and [`SKIPPED`].
    fragment: &'static str,
    /// The event the reader is expected to produce for [`Self::fragment`].
    event: fn() -> Event<'static>,
    /// Whether the reader must be configured with `check_end_names = false`.
    check_end_names: bool,
    /// Whether the reader must be configured with `allow_unmatched_ends = true`.
    allow_unmatched_ends: bool,
    /// Whether the trailing `<element/>` is still inside the namespace scope.
    /// This is the case only when [`Self::fragment`] itself opened a scope that
    /// the skipping method consumed instead of the one opened by `<root>`.
    trailing_is_bound: bool,
}

impl Scenario {
    /// The full document for this scenario.
    fn document(&self) -> String {
        format!("{PREFIX}{}{SKIPPED}</root><element/>", self.fragment)
    }

    /// The span [`read_to_end`](NsReader::read_to_end) must report: the content
    /// between the `>` of the opening tag and the `<` of the closing tag.
    fn span(&self) -> Range<u64> {
        let start = (PREFIX.len() + self.fragment.len()) as u64;
        start..start + SKIPPED.len() as u64
    }

    /// The namespace expected for the trailing `<element/>`.
    fn trailing_namespace(&self) -> ResolveResult<'static> {
        if self.trailing_is_bound {
            Bound(Namespace(b"namespace"))
        } else {
            Unbound
        }
    }

    /// Builds a reader for this scenario's document, applying its configuration.
    fn reader<'a>(&self, document: &'a str) -> NsReader<&'a [u8]> {
        let mut reader = NsReader::from_str(document);
        reader.config_mut().check_end_names = self.check_end_names;
        reader.config_mut().allow_unmatched_ends = self.allow_unmatched_ends;
        reader
    }

    /// The `Start` event of the opening `<root>` tag.
    fn root(&self) -> (ResolveResult<'static>, Event<'static>) {
        (
            Bound(Namespace(b"namespace")),
            Start(BytesStart::from_content("root xmlns='namespace'", 4)),
        )
    }
}

/// The ten scenarios, one per event kind that can precede the skipped content.
///
/// Some of them are not well-formed XML, but since the reader can parse them
/// they are kept as regression cases: they must not break the parser.
fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "decl",
            fragment: r#"<?xml version="1.0"?>"#,
            event: || Decl(BytesDecl::new("1.0", None, None)),
            check_end_names: true,
            allow_unmatched_ends: false,
            trailing_is_bound: false,
        },
        Scenario {
            name: "doctype",
            fragment: "<!DOCTYPE dtd>",
            event: || DocType(BytesText::new("dtd")),
            check_end_names: true,
            allow_unmatched_ends: false,
            trailing_is_bound: false,
        },
        Scenario {
            name: "pi",
            fragment: "<?pi?>",
            event: || PI(BytesPI::new("pi")),
            check_end_names: true,
            allow_unmatched_ends: false,
            trailing_is_bound: false,
        },
        Scenario {
            name: "comment",
            fragment: "<!--comment-->",
            event: || Comment(BytesText::new("comment")),
            check_end_names: true,
            allow_unmatched_ends: false,
            trailing_is_bound: false,
        },
        Scenario {
            name: "start",
            fragment: "<tag>",
            event: || Start(BytesStart::new("tag")),
            check_end_names: false,
            allow_unmatched_ends: false,
            // The unbalanced `<tag>` opened a scope which the skipping method
            // popped, so the `<root>` scope is still open.
            trailing_is_bound: true,
        },
        Scenario {
            name: "end",
            fragment: "</tag>",
            event: || End(BytesEnd::new("tag")),
            check_end_names: false,
            allow_unmatched_ends: true,
            trailing_is_bound: false,
        },
        Scenario {
            name: "empty",
            fragment: "<tag/>",
            event: || Empty(BytesStart::new("tag")),
            check_end_names: true,
            allow_unmatched_ends: false,
            trailing_is_bound: false,
        },
        Scenario {
            name: "text",
            fragment: "text",
            event: || Text(BytesText::new("text")),
            check_end_names: true,
            allow_unmatched_ends: false,
            trailing_is_bound: false,
        },
        Scenario {
            name: "cdata",
            fragment: "<![CDATA[cdata]]>",
            event: || CData(BytesCData::new("cdata")),
            check_end_names: true,
            allow_unmatched_ends: false,
            trailing_is_bound: false,
        },
        Scenario {
            name: "general_ref",
            fragment: "&entity;",
            event: || GeneralRef(BytesRef::new("entity")),
            check_end_names: true,
            allow_unmatched_ends: false,
            trailing_is_bound: false,
        },
    ]
}

/// Whether the fragment's event is one that the resolver assigns a namespace to.
/// `Start` and `Empty` push a scope, so they resolve inside `<root>`'s namespace.
fn fragment_namespace(event: &Event<'_>) -> ResolveResult<'static> {
    match event {
        Start(_) | Empty(_) | End(_) => Bound(Namespace(b"namespace")),
        _ => Unbound,
    }
}

#[test]
fn read_to_end() -> TestResult {
    for scenario in scenarios() {
        let document = scenario.document();
        let mut reader = scenario.reader(&document);
        let expected_event = (scenario.event)();

        assert_eq!(
            reader.read_resolved_event()?,
            scenario.root(),
            "{}",
            scenario.name
        );
        let (ns, event) = reader.read_resolved_event()?;
        assert_eq!(
            (ns, event),
            (fragment_namespace(&expected_event), expected_event),
            "{}",
            scenario.name,
        );
        assert_eq!(
            reader.read_to_end(QName(b"root"))?,
            scenario.span(),
            "{}",
            scenario.name,
        );
        assert_eq!(
            reader.read_resolved_event()?,
            (
                scenario.trailing_namespace(),
                Empty(BytesStart::new("element")),
            ),
            "{}",
            scenario.name,
        );
        assert_eq!(reader.read_event()?, Eof, "{}", scenario.name);
    }
    Ok(())
}

#[test]
fn read_to_end_into() -> TestResult {
    for scenario in scenarios() {
        let document = scenario.document();
        let mut reader = NsReader::from_reader(document.as_bytes());
        reader.config_mut().check_end_names = scenario.check_end_names;
        reader.config_mut().allow_unmatched_ends = scenario.allow_unmatched_ends;
        let expected_event = (scenario.event)();
        let buf = &mut Vec::new();

        {
            let (ns, event) = reader.read_resolved_event_into(buf)?;
            assert_eq!(
                (ns, event.into_owned()),
                scenario.root(),
                "{}",
                scenario.name,
            );
        }
        {
            let (ns, event) = reader.read_resolved_event_into(buf)?;
            assert_eq!(
                (ns, event.into_owned()),
                (fragment_namespace(&expected_event), expected_event),
                "{}",
                scenario.name,
            );
        }
        assert_eq!(
            reader.read_to_end_into(QName(b"root"), buf)?,
            scenario.span(),
            "{}",
            scenario.name,
        );
        {
            let (ns, event) = reader.read_resolved_event_into(buf)?;
            assert_eq!(
                (ns, event.into_owned()),
                (
                    scenario.trailing_namespace(),
                    Empty(BytesStart::new("element")),
                ),
                "{}",
                scenario.name,
            );
        }
        assert_eq!(reader.read_event_into(buf)?, Eof, "{}", scenario.name);
    }
    Ok(())
}

#[test]
fn read_text() -> TestResult {
    for scenario in scenarios() {
        let document = scenario.document();
        let mut reader = scenario.reader(&document);
        let expected_event = (scenario.event)();

        assert_eq!(
            reader.read_resolved_event()?,
            scenario.root(),
            "{}",
            scenario.name
        );
        let (ns, event) = reader.read_resolved_event()?;
        assert_eq!(
            (ns, event),
            (fragment_namespace(&expected_event), expected_event),
            "{}",
            scenario.name,
        );
        assert_eq!(
            reader.read_text(QName(b"root"))?,
            BytesText::from_escaped(SKIPPED),
            "{}",
            scenario.name,
        );
        assert_eq!(
            reader.read_resolved_event()?,
            (
                scenario.trailing_namespace(),
                Empty(BytesStart::new("element")),
            ),
            "{}",
            scenario.name,
        );
        assert_eq!(reader.read_event()?, Eof, "{}", scenario.name);
    }
    Ok(())
}

#[test]
fn read_text_into() -> TestResult {
    for scenario in scenarios() {
        let document = scenario.document();
        let mut reader = NsReader::from_reader(document.as_bytes());
        reader.config_mut().check_end_names = scenario.check_end_names;
        reader.config_mut().allow_unmatched_ends = scenario.allow_unmatched_ends;
        let expected_event = (scenario.event)();
        let buf = &mut Vec::new();

        {
            let (ns, event) = reader.read_resolved_event_into(buf)?;
            assert_eq!(
                (ns, event.into_owned()),
                scenario.root(),
                "{}",
                scenario.name,
            );
        }
        {
            let (ns, event) = reader.read_resolved_event_into(buf)?;
            assert_eq!(
                (ns, event.into_owned()),
                (fragment_namespace(&expected_event), expected_event),
                "{}",
                scenario.name,
            );
        }
        {
            let text = reader.read_text_into(QName(b"root"), buf)?;
            assert_eq!(text, BytesText::from_escaped(SKIPPED), "{}", scenario.name);
        }
        {
            let (ns, event) = reader.read_resolved_event_into(buf)?;
            assert_eq!(
                (ns, event.into_owned()),
                (
                    scenario.trailing_namespace(),
                    Empty(BytesStart::new("element")),
                ),
                "{}",
                scenario.name,
            );
        }
        assert_eq!(reader.read_event_into(buf)?, Eof, "{}", scenario.name);
    }
    Ok(())
}

/// `read_to_end` must skip nested elements that have the *literally* same name
/// as the one being closed, including when they carry namespace declarations.
#[test]
fn read_to_end_handles_nested_same_names() -> TestResult {
    let xml = "\
        <outer xmlns='namespace 1'>\
            <inner xmlns='namespace 2'>\
                <outer></outer>\
            </inner>\
            <inner>\
                <inner></inner>\
                <inner/>\
                <outer></outer>\
                <p:outer xmlns:p='ns'></p:outer>\
                <outer/>\
            </inner>\
        </outer>";
    let mut reader = NsReader::from_str(xml);

    let start = BytesStart::from_content("outer xmlns='namespace 1'", 5);
    assert_eq!(
        reader.read_resolved_event()?,
        (Bound(Namespace(b"namespace 1")), Start(start.clone())),
    );
    reader.read_to_end(start.to_end().name())?;
    assert_eq!(reader.read_resolved_event()?, (Unbound, Eof));
    Ok(())
}

/// A missing end tag is an ill-formed-document error rather than a silent EOF.
#[test]
fn read_to_end_without_end_tag_is_an_error() {
    let mut reader = NsReader::from_str("<root><inner>");
    assert!(matches!(reader.read_event(), Ok(Start(_))));
    assert!(matches!(
        reader.read_to_end(QName(b"root")),
        Err(Error::IllFormed(_)),
    ));
}

/// After `read_to_end` has consumed a subtree, the namespaces declared inside it
/// are gone, while the ones declared by its ancestors remain in scope.
#[test]
fn read_to_end_restores_the_enclosing_scope() -> TestResult {
    let mut reader =
        NsReader::from_str("<a xmlns:o='outer'><b xmlns:i='inner'><i:c/></b><o:d/><i:e/></a>");

    assert_eq!(reader.read_resolved_event()?.0, Unbound); // <a>
    let start = match reader.read_event()? {
        Start(e) => e.into_owned(),
        other => panic!("expected <b>, got {other:?}"),
    };
    reader.read_to_end(start.name())?;

    // `o` was declared by the still-open <a>.
    assert_eq!(reader.read_resolved_event()?.0, Bound(Namespace(b"outer")));
    // `i` went out of scope with <b>.
    assert_eq!(reader.read_resolved_event()?.0, Unknown(b"i".to_vec()));
    Ok(())
}
