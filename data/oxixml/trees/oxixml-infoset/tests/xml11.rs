//! [XML 1.1](https://www.w3.org/TR/xml11/) support in the tree the parser builds, and the
//! guarantee that XML 1.0 documents are unaffected by it.
//!
//! The two versions differ in exactly one production that matters to a data model, and they
//! differ in it twice over:
//!
//! * `Char` is wider in 1.1 — it admits every C0 control except `#x0` — so a *character
//!   reference* may denote one of them;
//! * `RestrictedChar` is subtracted from the `document` production, so those same characters may
//!   not be written *literally*.
//!
//! Line ends differ too: 1.1 folds `#x85` (NEL) and `#x2028` (LINE SEPARATOR) into `#xA`
//! alongside `#xD` and `#xD#xA`.
//!
//! The `Name` productions do *not* differ: XML 1.0 Fifth Edition adopted the 1.1 name characters
//! verbatim.
//!
//! # Why some of these differ from `oxixml-dom`
//!
//! This crate always runs the full well-formedness checker of `oxixml-xml`, which `oxixml-dom`
//! does not, so a document that violates a well-formedness constraint is rejected here rather
//! than being built into a tree. Where the two therefore disagree, the case below says so and
//! asserts what a conforming non-validating processor must do.

use oxixml_infoset::{Tree, XdmError, XdmNode};

const DECL_1_1: &str = "<?xml version=\"1.1\"?>";
const DECL_1_0: &str = "<?xml version=\"1.0\"?>";

fn text_of(xml: &str) -> Result<String, XdmError> {
    let tree = Tree::parse_str(xml)?;
    Ok(tree
        .document_element()
        .map(|root| root.string_value())
        .unwrap_or_default())
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
        assert!(
            text_of(&format!("{prolog}<doc>a&#x7;b</doc>")).is_err(),
            "`#x7` is not a `Char` in XML 1.0 ({prolog:?})"
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
    for body in [
        "a\u{7}b",
        "<!--a\u{7}b-->",
        "<![CDATA[a\u{7}b]]>",
        "<?target a\u{7}b?>",
    ] {
        let source = format!("{DECL_1_1}<doc>{body}</doc>");
        assert!(
            Tree::parse_str(&source).is_err(),
            "a literal `RestrictedChar` is subtracted from `document`: {body}"
        );
    }
    // In an attribute value it is just as fatal.
    assert!(Tree::parse_str(&format!("{DECL_1_1}<doc a=\"x\u{7}y\"/>")).is_err());
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
    // A CDATA section is character data too, so it is normalized on the same terms.
    assert_eq!(
        text_of(&format!(
            "{DECL_1_1}<doc><![CDATA[a\u{85}b\u{2028}c]]></doc>"
        ))
        .expect("well-formed XML 1.1"),
        "a\nb\nc"
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
    // End-of-line normalization happens to the characters of the entity before parsing; a
    // reference is not a character at that point.
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

#[test]
fn a_comment_and_a_processing_instruction_are_normalized_like_text() {
    let tree = Tree::parse_str(&format!(
        "{DECL_1_1}<doc><!--a\u{85}b--><?t c\u{2028}d?></doc>"
    ))
    .expect("well-formed XML 1.1");
    let children = tree.document_element().expect("root").children();
    assert_eq!(children[0].string_value(), "a\nb");
    assert_eq!(children[1].string_value(), "c\nd");
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

#[test]
fn names_are_the_same_in_both_versions() {
    // XML 1.0 Fifth Edition adopted the XML 1.1 `NameStartChar` / `NameChar` classes, so the
    // Dutch ligature `ĳ` and a non-BMP start character are names under both, and a digit is a
    // start character under neither.
    for prolog in [DECL_1_0, DECL_1_1] {
        assert!(
            Tree::parse_str(&format!("{prolog}<\u{133}oo x\u{133}=\"1\"/>")).is_ok(),
            "{prolog:?}"
        );
        assert!(
            Tree::parse_str(&format!("{prolog}<\u{10000}/>")).is_ok(),
            "{prolog:?}"
        );
        assert!(
            Tree::parse_str(&format!("{prolog}<0bad/>")).is_err(),
            "{prolog:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Attributes and entities
// ---------------------------------------------------------------------------

#[test]
fn an_attribute_value_may_name_a_restricted_character() -> Result<(), XdmError> {
    let tree = Tree::parse_str(&format!("{DECL_1_1}<doc att=\"form feed &#x0c;\"/>"))?;
    let attribute = tree
        .document_element()
        .and_then(|root| root.attributes().first().cloned())
        .expect("one attribute");
    assert_eq!(attribute.string_value(), "form feed \u{c}");
    Ok(())
}

#[test]
fn an_internal_entity_may_name_a_restricted_character() -> Result<(), XdmError> {
    let subset = "<!DOCTYPE doc [<!ENTITY bel \"&#x7;\">]>";
    assert_eq!(
        text_of(&format!("{DECL_1_1}{subset}<doc>a&bel;b</doc>"))?,
        "a\u{7}b"
    );
    // …and in an attribute value, which is expanded through a different path.
    let tree = Tree::parse_str(&format!("{DECL_1_1}{subset}<doc a=\"x&bel;y\"/>"))?;
    let attribute = tree
        .document_element()
        .and_then(|root| root.attributes().first().cloned())
        .expect("one attribute");
    assert_eq!(attribute.string_value(), "x\u{7}y");

    // Under 1.0 the same declaration is *ill-formed*, not merely inert: `&#x7;` inside an
    // `EntityValue` is a `Reference` to a character outside `Char`, which XML 1.0 §4.1 (WFC:
    // Legal Character) forbids. `oxixml-dom`, which does not run the well-formedness checker,
    // keeps the five characters as they were written instead of rejecting the document.
    assert!(Tree::parse_str(&format!("{subset}<doc>a&bel;b</doc>")).is_err());
    Ok(())
}

#[test]
fn a_doubly_escaped_reference_is_resolved_under_the_documents_version() -> Result<(), XdmError> {
    // Doubling the escape moves the reference from the declaration to the *use*: the replacement
    // text of `bel` is the five characters `&#x7;`, and XML 1.0 §4.4.2 ("Included") has them
    // processed "as though they were part of the document at the location the reference was
    // recognized" — so they are a character reference again, resolved under the document's own
    // version.
    let subset = "<!DOCTYPE doc [<!ENTITY bel \"&#38;#x7;\">]>";
    assert_eq!(
        text_of(&format!("{DECL_1_1}{subset}<doc>a&bel;b</doc>"))?,
        "a\u{7}b"
    );
    // Under 1.0 that second recognition is a reference to a character outside `Char`, so the
    // document is ill-formed rather than yielding the five characters literally. `oxixml-dom`,
    // which does not run the well-formedness checker, keeps them.
    assert!(Tree::parse_str(&format!("{DECL_1_0}{subset}<doc>a&bel;b</doc>")).is_err());
    Ok(())
}

#[test]
fn a_literal_restricted_character_in_the_internal_subset_is_rejected() -> Result<(), XdmError> {
    // The prolog is inside the `document` production too, and an entity declared there would
    // otherwise smuggle the character into a text node.
    let source = format!("{DECL_1_1}<!DOCTYPE r [<!ENTITY e \"a\u{7}b\">]><r>&e;</r>");
    assert!(Tree::parse_str(&source).is_err());
    // Written as a reference it is fine, and the entity expands to it.
    assert_eq!(
        text_of(&format!(
            "{DECL_1_1}<!DOCTYPE r [<!ENTITY e \"a&#x7;b\">]><r>&e;</r>"
        ))?,
        "a\u{7}b"
    );
    Ok(())
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
