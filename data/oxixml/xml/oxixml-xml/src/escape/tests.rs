use super::*;

#[test]
fn escape_specials() {
    assert_eq!(escape("<test>"), "&lt;test&gt;");
    assert_eq!(escape("a & b"), "a &amp; b");
    assert_eq!(escape("\"quote'"), "&quot;quote&apos;");
    // No special characters: borrowed, unchanged.
    assert!(matches!(escape("plain"), Cow::Borrowed("plain")));
}

#[test]
fn partial_and_minimal() {
    assert_eq!(partial_escape("<a> & \"x\""), "&lt;a&gt; &amp; \"x\"");
    assert_eq!(minimal_escape("<a> & \"x\""), "&lt;a> &amp; \"x\"");
}

#[test]
fn unescape_named() -> Result<(), EscapeError> {
    assert_eq!(unescape("&lt;test&gt;")?, "<test>");
    assert_eq!(unescape("a &amp; b")?, "a & b");
    assert_eq!(unescape("&quot;q&apos;")?, "\"q'");
    // No entities: borrowed.
    assert!(matches!(unescape("plain")?, Cow::Borrowed("plain")));
    Ok(())
}

#[test]
fn unescape_numeric() -> Result<(), EscapeError> {
    assert_eq!(unescape("&#48;")?, "0");
    assert_eq!(unescape("&#x30;")?, "0");
    assert_eq!(unescape("&#x1F600;")?, "\u{1F600}");
    assert_eq!(unescape("&#233;")?, "é");
    Ok(())
}

#[test]
fn escape_unescape_roundtrip() -> Result<(), EscapeError> {
    for original in [
        "plain text",
        "<tag attr=\"v\">body & 'more'</tag>",
        "ampersand & <less> >greater<",
        "quotes \" and ' mixed",
    ] {
        let escaped = escape(original);
        assert_eq!(
            unescape(&escaped)?,
            original,
            "roundtrip failed for {original:?}"
        );
    }
    Ok(())
}

#[test]
fn unescape_invalid_refs() {
    // Unterminated entity.
    assert_eq!(unescape("&lt"), Err(EscapeError::UnterminatedEntity(0..3)));
    // Unknown named entity.
    assert_eq!(
        unescape("&unknown;"),
        Err(EscapeError::UnrecognizedEntity(1..8, "unknown".to_string()))
    );
    // Numeric overflow / non-digits.
    assert!(matches!(
        unescape("&#xZZ;"),
        Err(EscapeError::InvalidCharRef(
            ParseCharRefError::InvalidNumber(_)
        ))
    ));
    // Signed number rejected.
    assert_eq!(
        unescape("&#x-1;"),
        Err(EscapeError::InvalidCharRef(
            ParseCharRefError::UnexpectedSign
        ))
    );
    // NUL is not permitted in XML.
    assert_eq!(
        unescape("&#0;"),
        Err(EscapeError::InvalidCharRef(
            ParseCharRefError::IllegalCharacter(0)
        ))
    );
    // Surrogate code point is not a valid scalar value.
    assert_eq!(
        unescape("&#xD800;"),
        Err(EscapeError::InvalidCharRef(
            ParseCharRefError::InvalidCodepoint(0xD800)
        ))
    );
}

#[test]
fn unescape_with_override() -> Result<(), EscapeError> {
    let resolver = |e: &str| match e {
        "custom" => Some("VALUE"),
        _ => resolve_xml_entity(e),
    };
    assert_eq!(unescape_with("&custom; &amp;", resolver)?, "VALUE &");
    Ok(())
}

#[test]
fn predefined_entities() {
    assert_eq!(resolve_xml_entity("lt"), Some("<"));
    assert_eq!(resolve_xml_entity("gt"), Some(">"));
    assert_eq!(resolve_xml_entity("amp"), Some("&"));
    assert_eq!(resolve_xml_entity("apos"), Some("'"));
    assert_eq!(resolve_xml_entity("quot"), Some("\""));
    assert_eq!(resolve_xml_entity("nbsp"), None);
}

#[cfg(not(feature = "escape-html"))]
#[test]
fn predefined_is_xml_only() {
    assert_eq!(resolve_predefined_entity("lt"), Some("<"));
    assert_eq!(resolve_predefined_entity("nbsp"), None);
}

#[cfg(feature = "escape-html")]
#[test]
fn predefined_is_html5() {
    assert_eq!(resolve_predefined_entity("lt"), Some("<"));
    assert_eq!(resolve_predefined_entity("nbsp"), Some("\u{A0}"));
    assert_eq!(resolve_html5_entity("amp"), Some("&"));
    assert_eq!(resolve_html5_entity("copy"), Some("\u{A9}"));
    assert_eq!(resolve_html5_entity("not-an-entity"), None);
}

#[test]
fn eol_normalization_xml10() {
    assert_eq!(normalize_xml10_eols("a\r\nb\rc\nd"), "a\nb\nc\nd");
    assert!(matches!(
        normalize_xml10_eols("no eol"),
        Cow::Borrowed("no eol")
    ));
}

#[test]
fn eol_normalization_xml11() {
    // \r\n, \r\x85, \x85, \u{2028}, lone \r all collapse to \n.
    assert_eq!(
        normalize_xml11_eols("a\r\nb\u{85}c\u{2028}d\re"),
        "a\nb\nc\nd\ne"
    );
}

#[test]
fn attribute_value_normalization() -> Result<(), EscapeError> {
    // Tabs, newlines and carriage returns become spaces; entities expand.
    assert_eq!(
        normalize_xml10_attribute_value("a\tb\r\nc &amp; d", 1, resolve_predefined_entity)?,
        "a b c & d"
    );
    // Already normalized: borrowed.
    assert!(matches!(
        normalize_xml10_attribute_value("plain", 1, resolve_predefined_entity)?,
        Cow::Borrowed("plain")
    ));
    Ok(())
}

/// The white-space rule of [XML 1.0 §3.3.3] applies to two different kinds of text, and the
/// difference is only visible on a `#xD#xA` pair.
///
/// The attribute's own value is end-of-line normalized first (step 1, §2.11), so a literal
/// `#xD#xA` there is a *single* line break and contributes a single space. An entity's
/// replacement text has been through the same normalization already, when its declaration was
/// parsed — so a `#xD` still present in it came from a character reference, is not a line break,
/// and contributes a space of its own. Character references written in the text that step 3
/// visits are not white space at all: "for a character reference, append the referenced
/// character".
///
/// [XML 1.0 §3.3.3]: https://www.w3.org/TR/xml/#AVNormalize
#[test]
fn attribute_value_normalization_of_line_breaks() -> Result<(), EscapeError> {
    // The replacement text an entity declaration yields: `&#13;&#10;` has become a real pair of
    // characters, while a literal line break in the declaration has become a single `#xA`.
    let resolver = |e: &str| match e {
        "charrefs" => Some("\r\n"),
        "literal" => Some("\n"),
        "quoted" => Some("&#13;"),
        _ => resolve_xml_entity(e),
    };

    // In the attribute's own value: one line break, one space.
    assert_eq!(
        normalize_xml10_attribute_value("a\r\nb", 4, resolver)?,
        "a b"
    );
    // In the attribute's own value: character references are appended as they stand.
    assert_eq!(
        normalize_xml10_attribute_value("a&#13;&#10;b", 4, resolver)?,
        "a\r\nb"
    );
    // In a replacement text: two white space characters, two spaces.
    assert_eq!(
        normalize_xml10_attribute_value("a&charrefs;b", 4, resolver)?,
        "a  b"
    );
    // In a replacement text: one white space character, one space.
    assert_eq!(
        normalize_xml10_attribute_value("a&literal;b", 4, resolver)?,
        "a b"
    );
    // In a replacement text: a character reference is still a character reference.
    assert_eq!(
        normalize_xml10_attribute_value("a&quoted;b", 4, resolver)?,
        "a\rb"
    );

    // XML 1.1 adds line breaks to the set the attribute's own value folds, and leaves the
    // per-character rule inside a replacement text untouched.
    assert_eq!(
        normalize_xml11_attribute_value("a\r\u{85}b", 4, resolver)?,
        "a b"
    );
    assert_eq!(
        normalize_xml11_attribute_value("a&charrefs;b", 4, resolver)?,
        "a  b"
    );
    Ok(())
}

#[test]
fn attribute_value_nested_entities_limit() {
    let resolver = |e: &str| match e {
        "a" => Some("&b;"),
        "b" => Some("&a;"),
        _ => resolve_xml_entity(e),
    };
    assert_eq!(
        normalize_xml10_attribute_value("&a;", 4, resolver),
        Err(EscapeError::TooManyNestedEntities)
    );
}
