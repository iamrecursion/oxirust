//! [XML 1.1](https://www.w3.org/TR/xml11/) support, and the guarantee that
//! XML 1.0 documents are unaffected by it.
//!
//! The two versions differ in exactly one production that matters to a document
//! model, and they differ in it twice over:
//!
//! * `Char` is wider in 1.1 — it admits every C0 control except `#x0` — so a
//!   *character reference* may denote one of them;
//! * `RestrictedChar` is subtracted from the `document` production, so those
//!   same characters may not be written *literally*.
//!
//! Line ends differ too: 1.1 folds `#x85` (NEL) and `#x2028` (LINE SEPARATOR)
//! into `#xA` alongside `#xD` and `#xD#xA`.
//!
//! The `Name` productions do *not* differ: XML 1.0 Fifth Edition adopted the
//! 1.1 name characters verbatim.

use oxixml_dom::build::{self, BuilderOptions};
use oxixml_dom::{BuildError, Document, DomError, SerializeOptions, XmlDeclaration};
use oxixml_xml::XmlVersion;

const DECL_1_1: &str = "<?xml version=\"1.1\"?>";
const DECL_1_0: &str = "<?xml version=\"1.0\"?>";

fn text_of(xml: &str) -> Result<String, BuildError> {
    let doc = build::parse_str(xml)?;
    let root = doc.document_element().expect("a document element");
    Ok(doc.text_content(root).unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Character references to the C0 controls
// ---------------------------------------------------------------------------

#[test]
fn xml11_character_references_may_denote_the_c0_controls() {
    // `#x7` (BEL) and `#xC` (FF) are `RestrictedChar`s: legal by reference.
    assert_eq!(
        text_of(&format!("{DECL_1_1}<doc>a&#x7;b&#x0c;c</doc>")).expect("well-formed XML 1.1"),
        "a\u{7}b\u{c}c"
    );
    // The whole restricted range, one document.
    let mut source = format!("{DECL_1_1}<doc>");
    let mut expected = String::new();
    for code in (0x1..=0x8_u32)
        .chain(0xB..=0xC)
        .chain(0xE..=0x1F)
        .chain(0x7F..=0x9F)
    {
        source.push_str(&format!("&#x{code:X};"));
        expected.push(char::from_u32(code).expect("a scalar value"));
    }
    source.push_str("</doc>");
    assert_eq!(text_of(&source).expect("well-formed XML 1.1"), expected);
}

#[test]
fn xml10_still_rejects_a_character_reference_to_a_c0_control() {
    for prolog in ["", DECL_1_0] {
        let error = text_of(&format!("{prolog}<doc>a&#x7;b</doc>"))
            .expect_err("`#x7` is not a `Char` in XML 1.0");
        assert!(
            matches!(error, BuildError::Dom(_)),
            "{prolog:?} gave {error}"
        );
    }
}

#[test]
fn neither_version_admits_a_reference_to_u0000() {
    for prolog in ["", DECL_1_0, DECL_1_1] {
        assert!(
            text_of(&format!("{prolog}<doc>&#0;</doc>")).is_err(),
            "`#x0` is outside `Char` in every version ({prolog:?})"
        );
    }
}

#[test]
fn a_restricted_character_written_literally_is_rejected_under_1_1() {
    let error = text_of(&format!("{DECL_1_1}<doc>a\u{7}b</doc>"))
        .expect_err("a literal `RestrictedChar` is subtracted from `document`");
    assert!(
        matches!(
            error,
            BuildError::RestrictedCharacter { character: '\u{7}' }
        ),
        "{error}"
    );
    // In an attribute value it is just as fatal, and the raw value is where the
    // provenance of the character is still visible.
    let error = build::parse_str(&format!("{DECL_1_1}<doc a=\"x\u{7}y\"/>"))
        .expect_err("a literal `RestrictedChar` in an attribute value");
    assert!(
        matches!(
            error,
            BuildError::RestrictedCharacter { character: '\u{7}' }
        ),
        "{error}"
    );
    // …and in a comment, a CDATA section and a processing instruction.
    for body in [
        "<!--a\u{7}b-->",
        "<![CDATA[a\u{7}b]]>",
        "<?target a\u{7}b?>",
    ] {
        let source = format!("{DECL_1_1}<doc>{body}</doc>");
        let error = build::parse_str(&source)
            .err()
            .unwrap_or_else(|| panic!("{body} must be rejected"));
        assert!(
            matches!(
                error,
                BuildError::RestrictedCharacter { character: '\u{7}' }
            ),
            "{body}: {error}"
        );
    }
}

#[test]
fn the_restricted_character_error_names_the_character() {
    let error = text_of(&format!("{DECL_1_1}<doc>\u{7}</doc>")).expect_err("literal `#x7`");
    let message = error.to_string();
    assert!(message.contains("U+0007"), "{message}");
    assert!(message.contains("character reference"), "{message}");
}

// ---------------------------------------------------------------------------
// Line ends
// ---------------------------------------------------------------------------

#[test]
fn xml11_normalizes_nel_and_line_separator_to_a_line_feed() {
    assert_eq!(
        text_of(&format!(
            "{DECL_1_1}<doc>a\u{85}b\u{2028}c\r\u{85}d\r\ne</doc>"
        ))
        .expect("well-formed XML 1.1"),
        "a\nb\nc\nd\ne"
    );
}

#[test]
fn xml10_leaves_nel_and_line_separator_alone() {
    for prolog in ["", DECL_1_0] {
        assert_eq!(
            text_of(&format!("{prolog}<doc>a\u{85}b\u{2028}c\r\nd</doc>"))
                .expect("well-formed XML 1.0"),
            "a\u{85}b\u{2028}c\nd",
            "{prolog:?}"
        );
    }
}

#[test]
fn a_character_reference_to_a_line_end_is_never_normalized() {
    // End-of-line normalization happens to the characters of the entity before
    // parsing; a reference is not a character at that point.
    for prolog in [DECL_1_0, DECL_1_1] {
        assert_eq!(
            text_of(&format!("{prolog}<doc>a&#xD;b&#xA;c</doc>")).expect("well-formed"),
            "a\rb\nc",
            "{prolog:?}"
        );
    }
    assert_eq!(
        text_of(&format!("{DECL_1_1}<doc>a&#x85;b&#x2028;c</doc>")).expect("well-formed XML 1.1"),
        "a\u{85}b\u{2028}c"
    );
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

#[test]
fn names_are_the_same_in_both_versions() {
    // XML 1.0 Fifth Edition adopted the XML 1.1 `NameStartChar` / `NameChar`
    // classes, so the Dutch ligature `ĳ` and a non-BMP start character are names
    // under both, and a digit is a start character under neither.
    for prolog in [DECL_1_0, DECL_1_1] {
        assert!(
            build::parse_str(&format!("{prolog}<\u{133}oo x\u{133}=\"1\"/>")).is_ok(),
            "{prolog:?}"
        );
        assert!(
            build::parse_str(&format!("{prolog}<\u{10000}/>")).is_ok(),
            "{prolog:?}"
        );
        assert!(
            build::parse_str(&format!("{prolog}<0bad/>")).is_err(),
            "{prolog:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Attributes and entities
// ---------------------------------------------------------------------------

#[test]
fn an_attribute_value_may_name_a_restricted_character() {
    let doc = build::parse_str(&format!("{DECL_1_1}<doc att=\"form feed &#x0c;\"/>"))
        .expect("well-formed XML 1.1");
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.get_attribute(root, "att"), Some("form feed \u{c}"));
}

#[test]
fn an_internal_entity_may_name_a_restricted_character() {
    let doc = build::parse_str(&format!(
        "{DECL_1_1}<!DOCTYPE doc [<!ENTITY bel \"&#x7;\">]><doc>a&bel;b</doc>"
    ))
    .expect("well-formed XML 1.1");
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.text_content(root).as_deref(), Some("a\u{7}b"));

    // Under 1.0 the same declaration leaves the reference unresolved, because
    // `&#x7;` is not a character reference there at all.
    let doc = build::parse_str("<!DOCTYPE doc [<!ENTITY bel \"&#x7;\">]><doc>a&bel;b</doc>")
        .expect("well-formed XML 1.0");
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.text_content(root).as_deref(), Some("a&#x7;b"));
}

// ---------------------------------------------------------------------------
// Serialization and round-tripping
// ---------------------------------------------------------------------------

#[test]
fn a_1_1_document_round_trips_through_the_serializer() {
    let source = format!("{DECL_1_1}<doc att=\"a&#x7;b\">c&#x1f;d&#x85;e</doc>");
    let doc = build::parse_str(&source).expect("well-formed XML 1.1");
    let xml = doc.to_xml();
    // The declaration is not optional for a 1.1 document: without it the result
    // would re-parse as 1.0, where none of these references is legal.
    assert!(xml.starts_with("<?xml version=\"1.1\"?>"), "{xml}");
    assert!(xml.contains("&#x7;"), "{xml}");
    assert!(xml.contains("&#x1F;"), "{xml}");
    assert!(xml.contains("&#x85;"), "{xml}");

    let reparsed = build::parse_str(&xml).expect("the serialized form is well-formed");
    assert!(
        doc.structural_eq(&reparsed),
        "{:?}",
        doc.structural_diff(&reparsed)
    );
    assert_eq!(reparsed.to_xml(), xml, "the serializer is a fixed point");
}

#[test]
fn a_1_0_document_is_serialized_exactly_as_before() {
    let doc = build::parse_str("<doc a=\"x&#9;y\">text &amp; more</doc>").expect("well-formed");
    assert_eq!(doc.to_xml(), "<doc a=\"x&#9;y\">text &amp; more</doc>");
    let doc = build::parse_str(&format!("{DECL_1_0}<doc/>")).expect("well-formed");
    assert_eq!(doc.to_xml(), "<doc/>");
    assert_eq!(
        doc.to_xml_with(&SerializeOptions::new().with_xml_declaration(XmlDeclaration::Include)),
        "<?xml version=\"1.0\"?>\n<doc/>"
    );
}

// ---------------------------------------------------------------------------
// Forcing a version
// ---------------------------------------------------------------------------

#[test]
fn the_version_can_be_forced_onto_a_document_without_a_declaration() {
    let options = BuilderOptions::new().with_xml_version(Some(XmlVersion::Explicit1_1));
    let doc = build::parse_str_with("<doc>a&#x7;b</doc>", &options).expect("forced XML 1.1");
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.text_content(root).as_deref(), Some("a\u{7}b"));
    // The document reports no declaration, because it had none…
    assert_eq!(doc.xml_version(), None);
    // …but the serializer still has to write one, or the output is not readable
    // back as the tree it came from.
    assert!(doc.to_xml().starts_with("<?xml version=\"1.1\"?>"));
}

#[test]
fn a_forced_version_outranks_the_declaration() {
    let options = BuilderOptions::new().with_xml_version(Some(XmlVersion::Explicit1_1));
    let doc = build::parse_str_with(&format!("{DECL_1_0}<doc>a&#x7;b</doc>"), &options)
        .expect("forced XML 1.1");
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.text_content(root).as_deref(), Some("a\u{7}b"));
    // The declaration itself is still reproduced as it was written.
    assert_eq!(doc.xml_version(), Some("1.0"));
}

// ---------------------------------------------------------------------------
// Tripwires: nothing about XML 1.0 changed
// ---------------------------------------------------------------------------

#[test]
fn xml10_documents_are_unchanged() {
    let cases: &[(&str, &str)] = &[
        ("<doc>plain</doc>", "plain"),
        ("<doc>a\r\nb\rc\nd</doc>", "a\nb\nc\nd"),
        ("<doc>a&#x9;b&#xA;c&#xD;d</doc>", "a\tb\nc\rd"),
        ("<doc>&#x10FFFF;</doc>", "\u{10FFFF}"),
        ("<doc>&#xFFFD;</doc>", "\u{FFFD}"),
        ("<doc><![CDATA[a\r\nb]]></doc>", "a\nb"),
        ("<doc>a\u{85}b</doc>", "a\u{85}b"),
        ("<doc>a\u{2028}b</doc>", "a\u{2028}b"),
    ];
    for (source, expected) in cases {
        assert_eq!(
            text_of(source).unwrap_or_else(|error| panic!("{source}: {error}")),
            *expected,
            "{source}"
        );
    }
    for source in [
        "<doc>a\u{c}b</doc>",
        "<doc>&#xFFFE;</doc>",
        "<doc>&#xD800;</doc>",
        "<doc>&#x110000;</doc>",
    ] {
        assert!(text_of(source).is_err(), "{source} must stay rejected");
    }
}

// ---------------------------------------------------------------------------
// The node factories, where no character reference is available
// ---------------------------------------------------------------------------

#[test]
fn only_text_and_attributes_may_hold_a_restricted_character() {
    let mut doc = Document::new();
    doc.set_xml_version(Some("1.1"));
    // Text and attribute values can name one with a character reference, so the
    // tree may hold it and the serializer writes the reference back.
    assert!(doc.create_text_node("a\u{7}b").is_ok());
    let attribute = doc.create_attribute("x").expect("an attribute node");
    assert!(doc.set_node_value(attribute, "a\u{7}b").is_ok());
    // A comment, a CDATA section and a processing instruction have no reference
    // syntax at all, so a restricted character could never be written back out.
    assert!(doc.create_comment("a\u{7}b").is_err());
    assert!(doc.create_cdata_section("a\u{7}b").is_err());
    assert!(doc.create_processing_instruction("t", "a\u{7}b").is_err());
    // …and nothing at all may hold one while the document is XML 1.0.
    let mut doc = Document::new();
    assert!(doc.create_text_node("a\u{7}b").is_err());
    assert!(doc.create_comment("a\u{7}b").is_err());
}

#[test]
fn a_literal_restricted_character_in_the_internal_subset_is_rejected() {
    // The prolog is inside the `document` production too, and an entity declared
    // there would otherwise smuggle the character into a text node.
    let source = format!("{DECL_1_1}<!DOCTYPE r [<!ENTITY e \"a\u{7}b\">]><r>&e;</r>");
    let error = build::parse_str(&source).expect_err("a literal `RestrictedChar` in the subset");
    assert!(
        matches!(
            error,
            BuildError::RestrictedCharacter { character: '\u{7}' }
        ),
        "{error}"
    );
    // Written as a reference it is fine, and the entity expands to it.
    let doc = build::parse_str(&format!(
        "{DECL_1_1}<!DOCTYPE r [<!ENTITY e \"a&#x7;b\">]><r>&e;</r>"
    ))
    .expect("well-formed XML 1.1");
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.text_content(root).as_deref(), Some("a\u{7}b"));
}

#[test]
fn importing_into_a_1_0_document_rejects_what_it_cannot_write() {
    let source =
        build::parse_str(&format!("{DECL_1_1}<doc>a&#x7;b</doc>")).expect("well-formed XML 1.1");
    let root = source.document_element().expect("a document element");

    // Into another XML 1.1 document the subtree imports unchanged.
    let mut target = Document::new();
    target.set_xml_version(Some("1.1"));
    assert!(target.import_node(&source, root, true).is_ok());

    // Into an XML 1.0 document it must not, or the result could not be
    // serialized: `&#x7;` is not a character reference there.
    let mut target = Document::new();
    let error = target
        .import_node(&source, root, true)
        .expect_err("the character has no XML 1.0 spelling");
    assert!(
        matches!(error, DomError::InvalidCharacter { .. }),
        "{error}"
    );
}
