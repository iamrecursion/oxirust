//! Integration tests that assert exact SAX callback sequences for
//! representative documents, plus locator-accuracy tests for the
//! [`ErrorHandler`].

use std::cell::RefCell;
use std::rc::Rc;

use oxixml_sax::{
    Attributes, ContentHandler, ErrorHandler, Feature, Handlers, LexicalHandler, SaxParseException,
    XMLReader,
};

/// A shared transcript of formatted event lines. [`ContentHandler`] and
/// [`LexicalHandler`] are separate slots in [`Handlers`] (so that unrelated
/// handler objects never need to alias each other's `&mut` access), so a test
/// that wants callbacks from *both* traits interleaved into one ordered log
/// funnels them through a shared, reference-counted, interior-mutable
/// [`EventLog`] instead of a single object implementing both traits directly.
#[derive(Clone, Default)]
struct EventLog(Rc<RefCell<Vec<String>>>);

impl EventLog {
    fn push(&self, event: String) {
        self.0.borrow_mut().push(event);
    }

    fn snapshot(&self) -> Vec<String> {
        self.0.borrow().clone()
    }
}

struct ContentRecorder(EventLog);

impl ContentHandler for ContentRecorder {
    fn start_document(&mut self) -> Result<(), oxixml_sax::SaxError> {
        self.0.push("StartDocument".to_string());
        Ok(())
    }

    fn end_document(&mut self) -> Result<(), oxixml_sax::SaxError> {
        self.0.push("EndDocument".to_string());
        Ok(())
    }

    fn start_prefix_mapping(
        &mut self,
        prefix: Option<&str>,
        uri: &str,
    ) -> Result<(), oxixml_sax::SaxError> {
        self.0
            .push(format!("StartPrefixMapping({:?}, {uri})", prefix));
        Ok(())
    }

    fn end_prefix_mapping(&mut self, prefix: Option<&str>) -> Result<(), oxixml_sax::SaxError> {
        self.0.push(format!("EndPrefixMapping({:?})", prefix));
        Ok(())
    }

    fn start_element(
        &mut self,
        uri: Option<&str>,
        local_name: &str,
        qname: &str,
        attributes: &Attributes,
    ) -> Result<(), oxixml_sax::SaxError> {
        let attrs: Vec<String> = attributes
            .iter()
            .map(|a| format!("{}={:?}", a.qname, a.value))
            .collect();
        self.0.push(format!(
            "StartElement({:?}, {local_name}, {qname}, [{}])",
            uri,
            attrs.join(", ")
        ));
        Ok(())
    }

    fn end_element(
        &mut self,
        uri: Option<&str>,
        local_name: &str,
        qname: &str,
    ) -> Result<(), oxixml_sax::SaxError> {
        self.0
            .push(format!("EndElement({:?}, {local_name}, {qname})", uri));
        Ok(())
    }

    fn characters(&mut self, text: &str) -> Result<(), oxixml_sax::SaxError> {
        self.0.push(format!("Characters({text:?})"));
        Ok(())
    }

    fn processing_instruction(
        &mut self,
        target: &str,
        data: &str,
    ) -> Result<(), oxixml_sax::SaxError> {
        self.0.push(format!("PI({target}, {data:?})"));
        Ok(())
    }

    fn skipped_entity(&mut self, name: &str) -> Result<(), oxixml_sax::SaxError> {
        self.0.push(format!("SkippedEntity({name})"));
        Ok(())
    }
}

struct LexicalRecorder(EventLog);

impl LexicalHandler for LexicalRecorder {
    fn comment(&mut self, text: &str) -> Result<(), oxixml_sax::SaxError> {
        self.0.push(format!("Comment({text:?})"));
        Ok(())
    }

    fn start_cdata(&mut self) -> Result<(), oxixml_sax::SaxError> {
        self.0.push("StartCData".to_string());
        Ok(())
    }

    fn end_cdata(&mut self) -> Result<(), oxixml_sax::SaxError> {
        self.0.push("EndCData".to_string());
        Ok(())
    }

    fn start_entity(&mut self, name: &str) -> Result<(), oxixml_sax::SaxError> {
        self.0.push(format!("StartEntity({name})"));
        Ok(())
    }

    fn end_entity(&mut self, name: &str) -> Result<(), oxixml_sax::SaxError> {
        self.0.push(format!("EndEntity({name})"));
        Ok(())
    }
}

fn record(xml: &str) -> Vec<String> {
    let log = EventLog::default();
    let mut content = ContentRecorder(log.clone());
    let mut lexical = LexicalRecorder(log.clone());
    {
        let mut handlers = Handlers::new()
            .with_content(&mut content)
            .with_lexical(&mut lexical);
        XMLReader::new()
            .parse_str(xml, &mut handlers)
            .expect("well-formed document parses");
    }
    log.snapshot()
}

#[test]
fn simple_element_with_attribute_and_text() {
    let events = record(r#"<greeting lang="en">hello</greeting>"#);
    assert_eq!(
        events,
        vec![
            "StartDocument".to_string(),
            r#"StartElement(None, greeting, greeting, [lang="en"])"#.to_string(),
            r#"Characters("hello")"#.to_string(),
            "EndElement(None, greeting, greeting)".to_string(),
            "EndDocument".to_string(),
        ]
    );
}

#[test]
fn nested_namespaces_and_prefix_mapping_scope() {
    let xml = r#"<a:root xmlns:a="urn:a"><b xmlns="urn:default"><a:leaf/></b></a:root>"#;
    let events = record(xml);
    assert_eq!(
        events,
        vec![
            "StartDocument".to_string(),
            r#"StartPrefixMapping(Some("a"), urn:a)"#.to_string(),
            r#"StartElement(Some("urn:a"), root, a:root, [])"#.to_string(),
            r#"StartPrefixMapping(None, urn:default)"#.to_string(),
            r#"StartElement(Some("urn:default"), b, b, [])"#.to_string(),
            r#"StartElement(Some("urn:a"), leaf, a:leaf, [])"#.to_string(),
            r#"EndElement(Some("urn:a"), leaf, a:leaf)"#.to_string(),
            r#"EndElement(Some("urn:default"), b, b)"#.to_string(),
            "EndPrefixMapping(None)".to_string(),
            r#"EndElement(Some("urn:a"), root, a:root)"#.to_string(),
            r#"EndPrefixMapping(Some("a"))"#.to_string(),
            "EndDocument".to_string(),
        ]
    );
}

#[test]
fn cdata_is_distinguished_from_plain_characters() {
    let xml = "<doc>plain <![CDATA[<raw> & unescaped]]> text</doc>";
    let events = record(xml);
    assert_eq!(
        events,
        vec![
            "StartDocument".to_string(),
            "StartElement(None, doc, doc, [])".to_string(),
            r#"Characters("plain ")"#.to_string(),
            "StartCData".to_string(),
            r#"Characters("<raw> & unescaped")"#.to_string(),
            "EndCData".to_string(),
            r#"Characters(" text")"#.to_string(),
            "EndElement(None, doc, doc)".to_string(),
            "EndDocument".to_string(),
        ]
    );
}

#[test]
fn comments_and_processing_instructions() {
    let xml = "<?xml version=\"1.0\"?>\n<?style href=\"a.css\"?><doc><!--note--></doc>";
    let events = record(xml);
    assert_eq!(
        events,
        vec![
            "StartDocument".to_string(),
            r#"Characters("\n")"#.to_string(),
            r#"PI(style, "href=\"a.css\"")"#.to_string(),
            "StartElement(None, doc, doc, [])".to_string(),
            r#"Comment("note")"#.to_string(),
            "EndElement(None, doc, doc)".to_string(),
            "EndDocument".to_string(),
        ]
    );
}

#[test]
fn mixed_content_interleaves_elements_and_text() {
    let xml = "<p>a<b>bold</b>c</p>";
    let events = record(xml);
    assert_eq!(
        events,
        vec![
            "StartDocument".to_string(),
            "StartElement(None, p, p, [])".to_string(),
            r#"Characters("a")"#.to_string(),
            "StartElement(None, b, b, [])".to_string(),
            r#"Characters("bold")"#.to_string(),
            "EndElement(None, b, b)".to_string(),
            r#"Characters("c")"#.to_string(),
            "EndElement(None, p, p)".to_string(),
            "EndDocument".to_string(),
        ]
    );
}

#[test]
fn predefined_entities_resolve_to_characters_others_are_skipped() {
    let xml = "<p>a &amp; b &#65; &custom; end</p>";
    let events = record(xml);
    assert_eq!(
        events,
        vec![
            "StartDocument".to_string(),
            "StartElement(None, p, p, [])".to_string(),
            r#"Characters("a ")"#.to_string(),
            r#"Characters("&")"#.to_string(),
            r#"Characters(" b ")"#.to_string(),
            r#"Characters("A")"#.to_string(),
            r#"Characters(" ")"#.to_string(),
            "SkippedEntity(custom)".to_string(),
            r#"Characters(" end")"#.to_string(),
            "EndElement(None, p, p)".to_string(),
            "EndDocument".to_string(),
        ]
    );
}

#[test]
fn empty_element_still_gets_matched_start_and_end() {
    let events = record("<a><b/></a>");
    assert_eq!(
        events,
        vec![
            "StartDocument".to_string(),
            "StartElement(None, a, a, [])".to_string(),
            "StartElement(None, b, b, [])".to_string(),
            "EndElement(None, b, b)".to_string(),
            "EndElement(None, a, a)".to_string(),
            "EndDocument".to_string(),
        ]
    );
}

#[test]
fn namespaces_feature_disabled_leaves_names_and_xmlns_attributes_untouched() {
    let log = EventLog::default();
    let mut recorder = ContentRecorder(log.clone());
    {
        let mut reader = XMLReader::new();
        reader.set_feature(Feature::Namespaces, false);
        let mut handlers = Handlers::new().with_content(&mut recorder);
        reader
            .parse_str(r#"<a:tag xmlns:a="urn:a" a:attr="v"/>"#, &mut handlers)
            .expect("parses without namespace processing");
    }
    assert_eq!(
        log.snapshot(),
        vec![
            "StartDocument".to_string(),
            r#"StartElement(None, a:tag, a:tag, [xmlns:a="urn:a", a:attr="v"])"#.to_string(),
            "EndElement(None, a:tag, a:tag)".to_string(),
            "EndDocument".to_string(),
        ]
    );
}

#[test]
fn namespace_prefixes_feature_retains_xmlns_attributes() {
    let log = EventLog::default();
    let mut recorder = ContentRecorder(log.clone());
    {
        let mut reader = XMLReader::new();
        reader.set_feature(Feature::NamespacePrefixes, true);
        let mut handlers = Handlers::new().with_content(&mut recorder);
        reader
            .parse_str(r#"<tag xmlns="urn:d" id="1"/>"#, &mut handlers)
            .expect("valid document");
    }
    assert_eq!(
        log.snapshot(),
        vec![
            "StartDocument".to_string(),
            r#"StartPrefixMapping(None, urn:d)"#.to_string(),
            r#"StartElement(Some("urn:d"), tag, tag, [xmlns="urn:d", id="1"])"#.to_string(),
            r#"EndElement(Some("urn:d"), tag, tag)"#.to_string(),
            "EndPrefixMapping(None)".to_string(),
            "EndDocument".to_string(),
        ]
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ErrorHandler / Locator accuracy
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Default)]
struct ErrorRecorder {
    fatal_errors: Vec<(String, u64, u64)>,
    errors: Vec<(String, u64, u64)>,
}

impl ErrorHandler for ErrorRecorder {
    fn error(&mut self, exception: &SaxParseException) -> Result<(), oxixml_sax::SaxError> {
        self.errors.push((
            exception.message().to_string(),
            exception.line_number(),
            exception.column_number(),
        ));
        Ok(())
    }

    fn fatal_error(&mut self, exception: &SaxParseException) -> Result<(), oxixml_sax::SaxError> {
        self.fatal_errors.push((
            exception.message().to_string(),
            exception.line_number(),
            exception.column_number(),
        ));
        Ok(())
    }
}

#[test]
fn fatal_error_reports_locator_at_unclosed_tag() {
    // Two lines of content: `<root>` on line 1, then five spaces and an
    // unclosed tag on line 2. Byte offsets on line 2: 5 spaces (columns 1-5),
    // then the `<` that starts the unclosed tag at column 6 — the exact
    // position `oxixml_xml::reader::Reader::error_position` reports for an
    // unclosed-tag `SyntaxError`.
    let xml = "<root>\n     <unclosed";
    let mut error_recorder = ErrorRecorder::default();
    let mut handlers = Handlers::new().with_error(&mut error_recorder);

    let result = XMLReader::new().parse_str(xml, &mut handlers);
    assert!(result.is_err(), "truncated document must fail to parse");

    assert_eq!(error_recorder.fatal_errors.len(), 1);
    let (message, line, column) = &error_recorder.fatal_errors[0];
    assert!(
        message.contains("not closed"),
        "expected an unclosed-tag message, got: {message}"
    );
    assert_eq!(*line, 2);
    assert_eq!(*column, 6);
    assert!(error_recorder.errors.is_empty());
}

/// Runs `parse_str` on a worker thread with a hard time budget, so that a
/// re-introduced non-terminating driver loop fails the test instead of hanging
/// the whole suite. Returns `Err(())` when the budget is exceeded.
fn parse_with_timeout(
    reader: XMLReader,
    xml: &'static str,
) -> std::result::Result<(bool, usize, usize), ()> {
    let (sender, receiver) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let mut error_recorder = ErrorRecorder::default();
        let is_err = {
            let mut handlers = Handlers::new().with_error(&mut error_recorder);
            reader.parse_str(xml, &mut handlers).is_err()
        };
        let _ = sender.send((
            is_err,
            error_recorder.errors.len(),
            error_recorder.fatal_errors.len(),
        ));
    });
    match receiver.recv_timeout(std::time::Duration::from_secs(30)) {
        Ok(outcome) => {
            let _ = worker.join();
            Ok(outcome)
        }
        Err(_) => Err(()),
    }
}

#[test]
fn ill_formed_error_is_fatal_and_aborts_parsing() {
    // XML 1.0 §1.2: a well-formedness violation is a *fatal* error, so a
    // mismatched end tag must stop the parse through `fatal_error`, not be
    // reported as a recoverable `error` while the event stream carries on
    // unbalanced.
    let xml = "<a><b></c></a>";
    let mut error_recorder = ErrorRecorder::default();
    let log = EventLog::default();
    let mut content = ContentRecorder(log.clone());
    let result = {
        let mut handlers = Handlers::new()
            .with_content(&mut content)
            .with_error(&mut error_recorder);
        XMLReader::new().parse_str(xml, &mut handlers)
    };
    assert!(result.is_err(), "a WF violation must abort parsing");
    assert!(error_recorder.errors.is_empty());
    assert_eq!(error_recorder.fatal_errors.len(), 1);
    let events = log.snapshot();
    assert!(events.contains(&"StartDocument".to_string()));
    assert!(
        !events.contains(&"EndDocument".to_string()),
        "endDocument must not be reported for an aborted parse"
    );
}

#[test]
fn mismatched_end_tag_terminates_with_well_formedness_checks() {
    // Regression: with `check_well_formedness` enabled, `oxixml_xml` reports an
    // `IllFormed` error without advancing its parse state, so a driver that
    // treated it as recoverable looped forever on this input.
    let mut reader = XMLReader::new();
    reader.xml_config_mut().check_well_formedness = true;
    let (is_err, errors, fatals) =
        parse_with_timeout(reader, "<a><b></a>").expect("parse must terminate promptly");
    assert!(is_err);
    assert_eq!(errors, 0);
    assert_eq!(fatals, 1);
}

#[test]
fn unclosed_element_terminates_with_well_formedness_checks() {
    let mut reader = XMLReader::new();
    reader.xml_config_mut().check_well_formedness = true;
    let (is_err, errors, fatals) =
        parse_with_timeout(reader, "<a><b>").expect("parse must terminate promptly");
    assert!(is_err);
    assert_eq!(errors, 0);
    assert_eq!(fatals, 1);
}

#[test]
fn unclosed_document_element_is_reported_by_default() {
    // No `check_well_formedness`: the driver's own end-of-entity check must
    // still catch the element left open.
    let mut error_recorder = ErrorRecorder::default();
    let log = EventLog::default();
    let mut content = ContentRecorder(log.clone());
    let result = {
        let mut handlers = Handlers::new()
            .with_content(&mut content)
            .with_error(&mut error_recorder);
        XMLReader::new().parse_str("<a><b/>", &mut handlers)
    };
    assert!(result.is_err(), "an unclosed root element must be reported");
    assert_eq!(error_recorder.fatal_errors.len(), 1);
    assert!(error_recorder.fatal_errors[0].0.contains("not closed"));
    assert!(!log.snapshot().contains(&"EndDocument".to_string()));
}

#[test]
fn character_data_after_the_document_element_is_reported() {
    let mut error_recorder = ErrorRecorder::default();
    let result = {
        let mut handlers = Handlers::new().with_error(&mut error_recorder);
        XMLReader::new().parse_str("<a>text</a> trailing", &mut handlers)
    };
    assert!(result.is_err());
    assert_eq!(error_recorder.fatal_errors.len(), 1);
    assert!(
        error_recorder.fatal_errors[0]
            .0
            .contains("outside the document element")
    );
}

#[test]
fn white_space_around_the_document_element_is_allowed() {
    let mut error_recorder = ErrorRecorder::default();
    let result = {
        let mut handlers = Handlers::new().with_error(&mut error_recorder);
        XMLReader::new().parse_str("\n <a>text</a> \n", &mut handlers)
    };
    assert!(result.is_ok(), "Misc white space is legal around the root");
    assert!(error_recorder.fatal_errors.is_empty());
}

/// Documents whose internal subset declares an entity whose value itself holds
/// references: `(name, internal subset, the reference as written, the character
/// data it stands for)`.
///
/// XML 1.0 §4.4 gives this a two-phase answer: a character reference inside an
/// `EntityValue` is resolved where the entity is *declared*, a general entity
/// reference where it is *used*, and neither pass rescans what it produced. The
/// same answer is required in content and in an attribute value, and
/// `oxixml-dom` and `oxixml-infoset` are held to this very table, so a document
/// means the same thing wherever the reference appears and whichever of the
/// three crates reads it.
const ENTITY_EXPANSIONS: &[(&str, &str, &str, &str)] = &[
    (
        "predefined-inside-the-value",
        r#"<!ENTITY e "bar &amp; baz">"#,
        "&e;",
        "bar & baz",
    ),
    (
        "character-reference-inside-the-value",
        r#"<!ENTITY copy "&#169;">"#,
        "&copy;",
        "\u{a9}",
    ),
    (
        "character-reference-in-running-text",
        r#"<!ENTITY nbsp "&#160;">"#,
        "x&nbsp;y",
        "x\u{a0}y",
    ),
    (
        "hexadecimal-character-reference",
        r#"<!ENTITY e "&#xe9;t&#xe9;">"#,
        "&e;",
        "\u{e9}t\u{e9}",
    ),
    (
        "nested-general-entity",
        r#"<!ENTITY inner "in"><!ENTITY outer "[&inner;]">"#,
        "&outer;",
        "[in]",
    ),
    (
        "escaped-markup-stays-text",
        r#"<!ENTITY tag "&lt;foo&gt;">"#,
        "&tag;",
        "<foo>",
    ),
    (
        // A `>` needs no escape anywhere, so the scanner that finds the end of
        // the declaration has to look past the one inside the quotes.
        "unescaped-greater-than-in-the-value",
        r#"<!ENTITY e "a > b">"#,
        "&e;",
        "a > b",
    ),
    (
        // A literal delimited by apostrophes may hold quotation marks, and a
        // quotation mark that reaches an attribute value through an entity is
        // data rather than the end of the literal (XML 1.0 §4.4, "Included in
        // Literal").
        "quotation-marks-in-an-apostrophe-quoted-value",
        r#"<!ENTITY e 'say "hi"'>"#,
        "&e;",
        "say \"hi\"",
    ),
    (
        "doubly-escaped-less-than",
        r#"<!ENTITY e "&#38;#60;">"#,
        "&e;",
        "<",
    ),
    (
        "doubly-escaped-ampersand",
        r#"<!ENTITY e "&#38;#38;">"#,
        "&e;",
        "&",
    ),
    (
        "triply-escaped-ampersand",
        r#"<!ENTITY e "&#38;#38;#38;">"#,
        "&e;",
        "&#38;",
    ),
    (
        "the-specifications-own-example",
        r#"<!ENTITY rights "All rights reserved"><!ENTITY book "La Peste: &#38;#38; 1947, &#38;rights;">"#,
        "&book;",
        "La Peste: & 1947, All rights reserved",
    ),
];

/// The document that puts `reference` in content.
fn in_content(subset: &str, reference: &str) -> String {
    format!("<!DOCTYPE r [{subset}]><r>{reference}</r>")
}

/// The document that puts `reference` in an attribute value.
fn in_attribute(subset: &str, reference: &str) -> String {
    format!("<!DOCTYPE r [{subset}]><r a=\"{reference}\"/>")
}

/// The values of every attribute reported to `start_element`.
fn attribute_values(xml: &str) -> Vec<String> {
    #[derive(Default)]
    struct Collect(Vec<String>);
    impl ContentHandler for Collect {
        fn start_element(
            &mut self,
            _uri: Option<&str>,
            _local_name: &str,
            _qname: &str,
            attributes: &Attributes,
        ) -> Result<(), oxixml_sax::SaxError> {
            self.0
                .extend(attributes.iter().map(|attribute| attribute.value.clone()));
            Ok(())
        }
    }

    let mut collect = Collect::default();
    {
        let mut handlers = Handlers::new().with_content(&mut collect);
        XMLReader::new()
            .parse_str(xml, &mut handlers)
            .expect("well-formed document parses");
    }
    collect.0
}

/// Concatenates everything reported through [`ContentHandler::characters`].
fn text_of(xml: &str) -> String {
    #[derive(Default)]
    struct Collect(String);
    impl ContentHandler for Collect {
        fn characters(&mut self, text: &str) -> Result<(), oxixml_sax::SaxError> {
            self.0.push_str(text);
            Ok(())
        }
    }

    let mut collect = Collect::default();
    {
        let mut handlers = Handlers::new().with_content(&mut collect);
        XMLReader::new()
            .parse_str(xml, &mut handlers)
            .expect("well-formed document parses");
    }
    collect.0
}

#[test]
fn declared_entities_expand_to_the_agreed_character_data() {
    for &(name, subset, reference, expected) in ENTITY_EXPANSIONS {
        assert_eq!(text_of(&in_content(subset, reference)), expected, "{name}");
    }
}

#[test]
fn an_attribute_value_expands_to_the_same_character_data_as_content() {
    for &(name, subset, reference, expected) in ENTITY_EXPANSIONS {
        assert_eq!(
            attribute_values(&in_attribute(subset, reference)),
            [expected],
            "{name}"
        );
    }
}

#[test]
fn an_expanded_entity_is_bracketed_by_the_lexical_handler() {
    let events = record(r#"<!DOCTYPE r [<!ENTITY e "text">]><r>&e;</r>"#);
    assert_eq!(
        events,
        vec![
            "StartDocument".to_string(),
            "StartElement(None, r, r, [])".to_string(),
            "StartEntity(e)".to_string(),
            r#"Characters("text")"#.to_string(),
            "EndEntity(e)".to_string(),
            "EndElement(None, r, r)".to_string(),
            "EndDocument".to_string(),
        ]
    );
}

#[test]
fn only_entities_without_replacement_text_are_skipped() {
    // Undeclared, external, and "the replacement text is markup" are the three
    // cases a SAX2 processor has no replacement text for.
    for (name, xml) in [
        ("undeclared", "<r>&undeclared;</r>"),
        (
            "external",
            r#"<!DOCTYPE r [<!ENTITY e SYSTEM "e.xml">]><r>&e;</r>"#,
        ),
        (
            "markup",
            r#"<!DOCTYPE r [<!ENTITY e "<b>x</b>">]><r>&e;</r>"#,
        ),
    ] {
        let events = record(xml);
        let expected = if name == "undeclared" {
            "SkippedEntity(undeclared)"
        } else {
            "SkippedEntity(e)"
        };
        assert!(
            events.iter().any(|event| event == expected),
            "{name}: {events:?}"
        );
        assert!(
            !events.iter().any(|event| event.starts_with("StartEntity(")),
            "{name}: a skipped entity is not bracketed: {events:?}"
        );
    }

    // A declared internal entity, by contrast, is never skipped.
    let events = record(r#"<!DOCTYPE r [<!ENTITY e "text">]><r>&e;</r>"#);
    assert!(
        !events
            .iter()
            .any(|event| event.starts_with("SkippedEntity(")),
        "{events:?}"
    );
}

#[test]
fn a_recursive_entity_is_rejected_instead_of_hanging() {
    for xml in [
        r#"<!DOCTYPE r [<!ENTITY e "&e;">]><r>&e;</r>"#,
        r#"<!DOCTYPE r [<!ENTITY a "&b;"><!ENTITY b "&a;">]><r>&a;</r>"#,
    ] {
        let mut handlers = Handlers::new();
        let error = XMLReader::new()
            .parse_str(xml, &mut handlers)
            .expect_err("a recursive entity must be rejected");
        assert!(
            matches!(error, oxixml_sax::SaxError::NotWellFormed(ref message) if message.contains("refers to itself")),
            "{error}"
        );
    }
}

#[test]
fn a_billion_laughs_declaration_is_refused() {
    let mut xml = String::from(r#"<!DOCTYPE r [<!ENTITY lol0 "lololololololololol">"#);
    for level in 1..10 {
        xml.push_str(&format!(
            r#"<!ENTITY lol{level} "{}">"#,
            format!("&lol{};", level - 1).repeat(10)
        ));
    }
    xml.push_str("]><r>&lol9;</r>");

    let mut handlers = Handlers::new();
    let error = XMLReader::new()
        .parse_str(&xml, &mut handlers)
        .expect_err("an exponential expansion must be refused");
    assert!(
        matches!(error, oxixml_sax::SaxError::NotWellFormed(_)),
        "{error}"
    );
}

#[test]
fn the_first_declaration_of_an_entity_binds() {
    assert_eq!(
        text_of(r#"<!DOCTYPE r [<!ENTITY e "first"><!ENTITY e "second">]><r>&e;</r>"#),
        "first"
    );
}

#[test]
fn a_reference_may_be_repeated_as_often_as_a_document_likes() {
    // Repeating a reference is not "amplification": what the entity stands for
    // is written out once per occurrence, so an ordinary page with tens of
    // thousands of `&nbsp;` in it must parse like any other.
    const OCCURRENCES: usize = 50_000;
    let mut xml = String::from(r#"<!DOCTYPE r [<!ENTITY nbsp "&#160;">]><r>"#);
    for _ in 0..OCCURRENCES {
        xml.push_str("&nbsp;");
    }
    xml.push_str("</r>");

    let text = text_of(&xml);
    assert_eq!(text.chars().count(), OCCURRENCES);
    assert!(text.chars().all(|character| character == '\u{a0}'));
}

/// The internal subset of a "billion laughs" declaration: ten levels, each
/// naming the previous one ten times, so the expansion of `&lol9;` is 10^9
/// copies of the innermost text while the document itself stays a few hundred
/// bytes long.
fn billion_laughs_subset() -> String {
    let mut subset = String::from(r#"<!ENTITY lol0 "lololololololololol">"#);
    for level in 1..10 {
        subset.push_str(&format!(
            r#"<!ENTITY lol{level} "{}">"#,
            format!("&lol{};", level - 1).repeat(10)
        ));
    }
    subset
}

#[test]
fn a_billion_laughs_attribute_value_is_refused_like_one_in_content() {
    // The attribute path expands through the underlying reader rather than
    // through this crate's own expander, so it needs its own proof that the
    // budget still bites: a *wide* declaration multiplies enormously without
    // ever nesting deeply enough to trip the depth limit alone.
    let subset = billion_laughs_subset();
    for xml in [
        format!("<!DOCTYPE r [{subset}]><r>&lol9;</r>"),
        format!("<!DOCTYPE r [{subset}]><r a=\"&lol9;\"/>"),
    ] {
        let mut handlers = Handlers::new();
        let error = XMLReader::new()
            .parse_str(&xml, &mut handlers)
            .expect_err("an exponential expansion must be refused");
        assert!(
            matches!(error, oxixml_sax::SaxError::NotWellFormed(_)),
            "{error}"
        );
    }
}
