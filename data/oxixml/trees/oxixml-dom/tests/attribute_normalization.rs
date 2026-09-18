//! [XML 1.0 §3.3.3](https://www.w3.org/TR/xml/#AVNormalize), attribute-value
//! normalization, pinned in both directions.
//!
//! The rule that is easy to get backwards, and that this file exists to keep
//! straight, is the difference between a white-space character that is *written*
//! in the attribute value and one that is *named by a character reference*:
//!
//! > * For a character reference, append the referenced character to the
//! >   normalized value.
//! > * For a white space character (#x20, #xD, #xA, #x9), append a space
//! >   character (#x20) to the normalized value.
//!
//! So `a="x&#10;y"` is `x`, a line feed, `y` — the reference *preserves* the line
//! feed — while `a="x⏎y"` written with a real line feed is `x`, a space, `y`.
//! Both are right; they simply are not the same document.
//!
//! Step 1 of the algorithm matters too: end-of-line normalization runs first, so
//! a literal `#xD#xA` is one line end and therefore *one* space, not two.
//!
//! Everything here is CDATA behaviour. The further collapsing that a *tokenized*
//! attribute type asks for needs the `<!ATTLIST>` type of the attribute, which
//! this crate only tracks for `ID`.

use oxixml_dom::build;

fn attribute(xml: &str, name: &str) -> String {
    let doc = build::parse_str(xml).unwrap_or_else(|error| panic!("{xml}: {error}"));
    let root = doc.document_element().expect("a document element");
    doc.get_attribute(root, name)
        .unwrap_or_else(|| panic!("{xml}: no attribute {name}"))
        .to_owned()
}

#[test]
fn literal_white_space_becomes_a_space() {
    assert_eq!(attribute("<r a=\"x\ny\"/>", "a"), "x y");
    assert_eq!(attribute("<r a=\"x\ty\"/>", "a"), "x y");
    assert_eq!(attribute("<r a=\"x\ry\"/>", "a"), "x y");
    assert_eq!(attribute("<r a=\"x y\"/>", "a"), "x y");
}

#[test]
fn a_literal_carriage_return_line_feed_pair_is_one_space() {
    // §3.3.3 step 1 folds `#xD#xA` into a single `#xA` before step 3 turns white
    // space into a space, so the pair contributes one space and not two.
    assert_eq!(attribute("<r a=\"x\r\ny\"/>", "a"), "x y");
    assert_eq!(attribute("<r a=\"x\r\n\r\ny\"/>", "a"), "x  y");
    assert_eq!(attribute("<r a=\"x\n\ry\"/>", "a"), "x  y");
}

#[test]
fn a_character_reference_preserves_the_character_it_names() {
    assert_eq!(attribute("<r a=\"x&#10;y\"/>", "a"), "x\ny");
    assert_eq!(attribute("<r a=\"x&#13;y\"/>", "a"), "x\ry");
    assert_eq!(attribute("<r a=\"x&#9;y\"/>", "a"), "x\ty");
    assert_eq!(attribute("<r a=\"x&#32;y\"/>", "a"), "x y");
    // Adjacent references stay two characters: no end-of-line normalization is
    // applied to them, so `&#13;&#10;` is not folded into one.
    assert_eq!(attribute("<r a=\"x&#13;&#10;y\"/>", "a"), "x\r\ny");
}

#[test]
fn a_trailing_space_and_a_trailing_reference_are_both_kept() {
    // CDATA normalization neither trims nor collapses: what step 3 produces is
    // the value, trailing white space included.
    assert_eq!(attribute("<r a=\"a \"/>", "a"), "a ");
    assert_eq!(attribute("<r a=\"a&#10;\"/>", "a"), "a\n");
    assert_eq!(attribute("<r a=\"a&#32;\"/>", "a"), "a ");
    assert_eq!(attribute("<r a=\"  a  b  \"/>", "a"), "  a  b  ");
}

#[test]
fn white_space_reached_through_an_entity_becomes_a_space() {
    // Step 3 recurses into an entity's replacement text, and the characters it
    // finds there are ordinary characters — the character reference that put
    // them there was resolved when the entity was *declared* (§4.5), not here.
    // The specification's own table says `a="&d;&d;A&a;&#x20;&a;B&da;"` is
    // `#x20#x20A#x20#x20#x20B…`, i.e. entity-borne white space collapses while
    // the directly written `&#x20;` does not.
    let subset = "<!DOCTYPE r [<!ENTITY nl \"&#10;\"><!ENTITY lit \"\n\">]>";
    assert_eq!(attribute(&format!("{subset}<r a=\"x&nl;y\"/>"), "a"), "x y");
    assert_eq!(
        attribute(&format!("{subset}<r a=\"x&lit;y\"/>"), "a"),
        "x y"
    );
}

#[test]
fn the_predefined_entities_are_not_white_space() {
    assert_eq!(
        attribute("<r a=\"&lt;&amp;&gt;&quot;&apos;\"/>", "a"),
        "<&>\"'"
    );
}

#[test]
fn a_namespace_declaration_is_normalized_like_any_other_attribute() {
    // A namespace declaration is an ordinary attribute in this model, and its
    // value goes through the same normalization: the namespace name is what
    // §3.3.3 produces, not the bytes between the quotes.
    assert_eq!(attribute("<r xmlns:p=\"urn:a\nb\"/>", "xmlns:p"), "urn:a b");
    assert_eq!(attribute("<r xmlns=\"urn:a\tb\"/>", "xmlns"), "urn:a b");
    assert_eq!(
        attribute("<r xmlns:p=\"urn:a&#10;b\"/>", "xmlns:p"),
        "urn:a\nb"
    );
}
