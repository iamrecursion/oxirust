//! Unit tests for the optional well-formedness checks.

use crate::errors::{Error, WellFormedError};
use crate::events::Event;
use crate::reader::Reader;

/// Parses `xml` with all checks enabled and returns the first error, if any.
fn parse(xml: &str) -> Result<(), Error> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().enable_all_checks(true);
    loop {
        match reader.read_event()? {
            Event::Eof => return Ok(()),
            _ => continue,
        }
    }
}

/// Asserts that `xml` is rejected and returns the reported error.
fn reject(xml: &str) -> Error {
    match parse(xml) {
        Ok(()) => panic!("`{xml}` was accepted but is not well-formed"),
        Err(error) => error,
    }
}

#[test]
fn accepts_well_formed_documents() -> Result<(), Error> {
    parse("<doc/>")?;
    parse("<?xml version='1.0' encoding='UTF-8' standalone='yes'?>\n<doc>text</doc>\n")?;
    parse("<!-- lead --><?pi data?><doc><a/><![CDATA[<not markup>]]></doc><!-- trail -->")?;
    parse("<!DOCTYPE doc [<!ELEMENT doc (#PCDATA)><!ENTITY e 'text'>]><doc>&e;&amp;</doc>")?;
    parse("<doc a='&#65;&amp;' b=\"v\"/>")?;
    Ok(())
}

#[test]
fn rejects_structural_problems() {
    assert!(matches!(
        reject(""),
        Error::NotWellFormed(WellFormedError::MissingRootElement)
    ));
    assert!(matches!(
        reject("<a/><b/>"),
        Error::NotWellFormed(WellFormedError::MultipleRootElements)
    ));
    assert!(matches!(
        reject("<a/>trailing"),
        Error::NotWellFormed(WellFormedError::ContentOutsideRoot)
    ));
    assert!(matches!(
        reject("<a/><![CDATA[x]]>"),
        Error::NotWellFormed(WellFormedError::ContentOutsideRoot)
    ));
    assert!(matches!(
        reject(" <?xml version='1.0'?><a/>"),
        Error::NotWellFormed(WellFormedError::MisplacedXmlDeclaration)
    ));
    assert!(matches!(
        reject("<a><!DOCTYPE doc></a>"),
        Error::NotWellFormed(WellFormedError::MisplacedDoctype)
    ));
}

#[test]
fn rejects_names_and_characters() {
    assert!(matches!(
        reject("<0a/>"),
        Error::NotWellFormed(WellFormedError::InvalidName(_))
    ));
    assert!(matches!(
        reject("<doc>\u{c}</doc>"),
        Error::NotWellFormed(WellFormedError::InvalidCharacter('\u{c}'))
    ));
    assert!(matches!(
        reject("<doc>]]></doc>"),
        Error::NotWellFormed(WellFormedError::CDataEndInContent)
    ));
    assert!(matches!(
        reject("<doc><?xml-stylesheet?><?XML?></doc>"),
        Error::NotWellFormed(WellFormedError::ReservedPiTarget(_))
    ));
}

#[test]
fn rejects_attribute_problems() {
    assert!(matches!(
        reject("<doc a='<'/>"),
        Error::NotWellFormed(WellFormedError::LessThanInAttributeValue(_))
    ));
    assert!(matches!(
        reject("<doc a='&#123:'/>"),
        Error::NotWellFormed(WellFormedError::InvalidReference(_))
    ));
    assert!(matches!(
        reject("<doc x='1' x='2'/>"),
        Error::NotWellFormed(WellFormedError::DuplicateAttribute(_))
    ));
    assert!(matches!(
        reject("<doc a='1'b='2'/>"),
        Error::NotWellFormed(WellFormedError::MalformedTag(_))
    ));
}

#[test]
fn rejects_entity_problems() {
    assert!(matches!(
        reject("<doc>&foo;</doc>"),
        Error::NotWellFormed(WellFormedError::UndeclaredEntity(_))
    ));
    assert!(matches!(
        reject("<!DOCTYPE doc [<!ENTITY e1 '&e2;'><!ENTITY e2 '&e1;'>]><doc>&e1;</doc>"),
        Error::NotWellFormed(WellFormedError::IllFormedEntity { .. })
    ));
    assert!(matches!(
        reject("<!DOCTYPE doc [<!ENTITY e SYSTEM 'x' NDATA n>]><doc>&e;</doc>"),
        Error::NotWellFormed(WellFormedError::UnparsedEntityReference(_))
    ));
    assert!(matches!(
        reject("<!DOCTYPE doc [<!ENTITY e '&#60;foo>'>]><doc>&e;</doc>"),
        Error::NotWellFormed(WellFormedError::IllFormedEntity { .. })
    ));
}

#[test]
fn default_configuration_stays_permissive() -> Result<(), Error> {
    // Everything above is accepted by the `quick-xml`-compatible default path.
    for xml in [
        "<a/><b/>",
        "<0a/>",
        "<doc>&foo;</doc>",
        "<doc a='<'/>",
        "trailing text",
    ] {
        let mut reader = Reader::from_str(xml);
        loop {
            if matches!(reader.read_event()?, Event::Eof) {
                break;
            }
        }
    }
    Ok(())
}

/// Parses `xml` with all checks enabled through the namespace-aware reader.
fn parse_ns(xml: &str) -> Result<(), Error> {
    let mut reader = crate::reader::NsReader::from_str(xml);
    reader.config_mut().enable_all_checks(true);
    loop {
        match reader.read_event()? {
            Event::Eof => return Ok(()),
            _ => continue,
        }
    }
}

/// Every code point of the XML 1.1 `RestrictedChar` production.
const RESTRICTED_CHARS: [char; 10] = [
    '\u{1}', '\u{8}', '\u{B}', '\u{C}', '\u{E}', '\u{1F}', '\u{7F}', '\u{84}', '\u{86}', '\u{9F}',
];

/// XML 1.1 section 2.1 subtracts `RestrictedChar` from the `document`
/// production: those characters are a fatal error when they appear literally.
///
/// Covers the W3C cases `ibm-1-1-not-wf-P02-ibm02n01`..`n63` and
/// `eduni/xml-1.1/rmt-011`, `rmt-013`, `rmt-039` and `rmt-041`.
#[test]
fn rejects_literal_xml11_restricted_characters() {
    for ch in RESTRICTED_CHARS {
        for xml in [
            format!("<?xml version='1.1'?><root>{ch}</root>"),
            format!("<?xml version='1.1'?><root a='{ch}'/>"),
            format!("<?xml version='1.1'?><root><![CDATA[{ch}]]></root>"),
            format!("<?xml version='1.1'?><!-- {ch} --><root/>"),
            format!("<?xml version='1.1'?><!DOCTYPE root [<!ELEMENT root{ch} EMPTY>]><root/>"),
        ] {
            assert!(
                matches!(
                    reject(&xml),
                    Error::NotWellFormed(
                        WellFormedError::InvalidCharacter(_) | WellFormedError::InvalidDoctype(_)
                    )
                ),
                "U+{:04X} was accepted in `{}`",
                ch as u32,
                xml.escape_debug(),
            );
        }
    }
}

/// A `RestrictedChar` remains legal when it is written as a character
/// reference, and the characters that are *not* restricted stay legal too.
#[test]
fn accepts_referenced_xml11_restricted_characters() -> Result<(), Error> {
    for ch in RESTRICTED_CHARS {
        parse(&format!(
            "<?xml version='1.1'?><root>&#{};</root>",
            ch as u32
        ))?;
        parse(&format!(
            "<?xml version='1.1'?><!DOCTYPE root [<!ENTITY e '&#{};'>]><root>&e;</root>",
            ch as u32
        ))?;
    }
    // `#x85` and `#x2028` are line endings in XML 1.1, not restricted characters.
    parse("<?xml version='1.1'?><root>\u{85}\u{2028}</root>")?;
    Ok(())
}

/// The restriction applies only to XML 1.1: in XML 1.0 the very same code
/// points are either plain characters (`#x85`) or already outside `Char`.
#[test]
fn xml10_character_rules_are_unchanged() -> Result<(), Error> {
    parse("<?xml version='1.0'?><root>\u{85}</root>")?;
    assert!(matches!(
        reject("<?xml version='1.0'?><root>\u{1}</root>"),
        Error::NotWellFormed(WellFormedError::InvalidCharacter('\u{1}'))
    ));
    Ok(())
}

/// Namespaces in XML 1.1 section 6.1 allows `xmlns:p=""` to undeclare a prefix,
/// which Namespaces in XML 1.0 forbids.
///
/// Covers the W3C cases `eduni/namespaces/1.1/rmt-ns11-003` and `-004`.
#[test]
fn xml11_allows_prefix_undeclaration() -> Result<(), Error> {
    parse_ns("<?xml version='1.1'?><foo xmlns:a='http://example.org/ns'><bar xmlns:a=''/></foo>")?;
    // ... but XML 1.0 does not.
    let mut reader = crate::reader::NsReader::from_str(
        "<?xml version='1.0'?><foo xmlns:a='http://example.org/ns'><bar xmlns:a=''/></foo>",
    );
    reader.config_mut().enable_all_checks(true);
    let error = loop {
        match reader.read_event() {
            Ok(Event::Eof) => panic!("prefix undeclaration must be rejected in XML 1.0"),
            Ok(_) => continue,
            Err(error) => break error,
        }
    };
    assert!(matches!(
        error,
        Error::NotWellFormed(WellFormedError::EmptyPrefixBinding(_))
    ));
    // Undeclaring the *default* namespace is allowed in both versions.
    parse_ns("<?xml version='1.0'?><foo xmlns='http://example.org/ns'><bar xmlns=''/></foo>")?;
    Ok(())
}

/// The Entity Declared constraint is suspended by an external subset only when
/// the document is not `standalone="yes"` (XML 1.0 section 4.1).
#[test]
fn standalone_documents_keep_the_entity_declared_constraint() -> Result<(), Error> {
    assert!(matches!(
        reject("<?xml version='1.0' standalone='yes'?><!DOCTYPE d SYSTEM 'd.dtd'><d>&e;</d>"),
        Error::NotWellFormed(WellFormedError::UndeclaredEntity(_))
    ));
    assert!(matches!(
        reject(
            "<?xml version='1.0' standalone='yes'?><!DOCTYPE d [<!ENTITY % p SYSTEM 'p'>%p;]><d>&e;</d>"
        ),
        Error::NotWellFormed(WellFormedError::UndeclaredEntity(_))
    ));
    assert!(matches!(
        reject("<?xml version='1.0' standalone='yes'?><!DOCTYPE d SYSTEM 'd.dtd'><d a='&e;'/>"),
        Error::NotWellFormed(WellFormedError::UndeclaredEntity(_))
    ));
    // Without `standalone="yes"` the declaration may live in the external subset.
    parse("<?xml version='1.0' standalone='no'?><!DOCTYPE d SYSTEM 'd.dtd'><d>&e;</d>")?;
    parse("<?xml version='1.0'?><!DOCTYPE d SYSTEM 'd.dtd'><d>&e;</d>")?;
    Ok(())
}

/// `Comment ::= '<!--' ((Char - '-') | ('-' (Char - '-')))* '-->'` cannot match
/// a body that ends with `-`, inside the internal subset just as anywhere else.
#[test]
fn rejects_comment_ending_with_hyphen_inside_the_dtd() {
    assert!(matches!(
        reject("<!DOCTYPE doc [<!----->]><doc/>"),
        Error::NotWellFormed(WellFormedError::InvalidDoctype(_))
    ));
    assert!(matches!(
        reject("<!DOCTYPE doc [<!--a--b-->]><doc/>"),
        Error::NotWellFormed(WellFormedError::InvalidDoctype(_))
    ));
    // The same comment in element content is rejected as well.
    assert!(matches!(
        reject("<doc><!-----></doc>"),
        Error::IllFormed(crate::errors::IllFormedError::DoubleHyphenInComment)
    ));
}

/// The "billion laughs" bomb never repeats an entity name and stays far below
/// the nesting limit, so only the expansion budget stops it.
#[test]
fn rejects_exponential_entity_expansion() {
    let mut dtd = String::from("<!DOCTYPE bomb [<!ENTITY lol0 'lol'>");
    for level in 1..=9 {
        dtd.push_str(&format!(
            "<!ENTITY lol{level} '{}'>",
            format!("&lol{};", level - 1).repeat(10)
        ));
    }
    dtd.push_str("]>");

    let content = format!("{dtd}<bomb>&lol9;</bomb>");
    assert!(matches!(
        reject(&content),
        Error::NotWellFormed(WellFormedError::EntityExpansionLimit { .. })
    ));

    let attribute = format!("{dtd}<bomb a='&lol9;'/>");
    assert!(matches!(
        reject(&attribute),
        Error::NotWellFormed(WellFormedError::EntityExpansionLimit { .. })
    ));

    // A moderate amount of expansion is still accepted.
    let mut small = String::from("<!DOCTYPE ok [<!ENTITY e0 'text'>");
    for level in 1..=3 {
        small.push_str(&format!(
            "<!ENTITY e{level} '{}'>",
            format!("&e{};", level - 1).repeat(3)
        ));
    }
    small.push_str("]><ok>&e3;</ok>");
    assert!(parse(&small).is_ok(), "a small expansion must be accepted");
}
