//! [XML 1.1](https://www.w3.org/TR/xml11/) behaviour of the SAX reader, and the
//! attribute-value normalization of
//! [XML 1.0 §3.3.3](https://www.w3.org/TR/xml/#AVNormalize), pinned so that this
//! crate and `oxixml-dom` cannot drift apart: a document that declares entities
//! and white space has to mean the same thing whichever of the two reads it.

use oxixml_sax::{Attributes, ContentHandler, Handlers, SaxError, XMLReader};

/// Collects every `characters` callback into one string, and the attributes of
/// the first element into a list.
#[derive(Default)]
struct Collector {
    text: String,
    attributes: Vec<(String, String)>,
}

impl ContentHandler for Collector {
    fn start_element(
        &mut self,
        _namespace: Option<&str>,
        _local_name: &str,
        _qualified_name: &str,
        attributes: &Attributes,
    ) -> Result<(), SaxError> {
        if self.attributes.is_empty() {
            for attribute in attributes.iter() {
                self.attributes
                    .push((attribute.qname.clone(), attribute.value.clone()));
            }
        }
        Ok(())
    }

    fn characters(&mut self, text: &str) -> Result<(), SaxError> {
        self.text.push_str(text);
        Ok(())
    }
}

fn collect(xml: &str) -> Collector {
    let mut collector = Collector::default();
    {
        let mut handlers = Handlers::new().with_content(&mut collector);
        XMLReader::new()
            .parse_str(xml, &mut handlers)
            .unwrap_or_else(|error| panic!("{xml}: {error}"));
    }
    collector
}

fn attribute(xml: &str, name: &str) -> String {
    collect(xml)
        .attributes
        .into_iter()
        .find(|(key, _)| key == name)
        .unwrap_or_else(|| panic!("{xml}: no attribute {name}"))
        .1
}

const DECL_1_1: &str = "<?xml version=\"1.1\"?>";

// ---------------------------------------------------------------------------
// Attribute-value normalization: literal white space versus a reference
// ---------------------------------------------------------------------------

#[test]
fn literal_white_space_becomes_a_space() {
    assert_eq!(attribute("<r a=\"x\ny\"/>", "a"), "x y");
    assert_eq!(attribute("<r a=\"x\ty\"/>", "a"), "x y");
    assert_eq!(attribute("<r a=\"x\ry\"/>", "a"), "x y");
}

#[test]
fn a_literal_carriage_return_line_feed_pair_is_one_space() {
    assert_eq!(attribute("<r a=\"x\r\ny\"/>", "a"), "x y");
    assert_eq!(attribute("<r a=\"x\r\n\r\ny\"/>", "a"), "x  y");
}

#[test]
fn a_character_reference_preserves_the_character_it_names() {
    assert_eq!(attribute("<r a=\"x&#10;y\"/>", "a"), "x\ny");
    assert_eq!(attribute("<r a=\"x&#13;y\"/>", "a"), "x\ry");
    assert_eq!(attribute("<r a=\"x&#9;y\"/>", "a"), "x\ty");
    assert_eq!(attribute("<r a=\"x&#13;&#10;y\"/>", "a"), "x\r\ny");
}

#[test]
fn a_trailing_space_and_a_trailing_reference_are_both_kept() {
    assert_eq!(attribute("<r a=\"a \"/>", "a"), "a ");
    assert_eq!(attribute("<r a=\"a&#10;\"/>", "a"), "a\n");
}

#[test]
fn white_space_reached_through_an_entity_becomes_a_space() {
    let subset = "<!DOCTYPE r [<!ENTITY nl \"&#10;\">]>";
    assert_eq!(attribute(&format!("{subset}<r a=\"x&nl;y\"/>"), "a"), "x y");
}

// ---------------------------------------------------------------------------
// XML 1.1
// ---------------------------------------------------------------------------

#[test]
fn an_internal_entity_may_name_a_restricted_character_under_1_1() {
    let subset = "<!DOCTYPE r [<!ENTITY bel \"&#x7;\">]>";
    assert_eq!(
        collect(&format!("{DECL_1_1}{subset}<r>a&bel;b</r>")).text,
        "a\u{7}b"
    );
    // Under XML 1.0 `&#x7;` is not a character reference, so it is not resolved
    // when the entity is declared and stays in the replacement text verbatim.
    assert_eq!(collect(&format!("{subset}<r>a&bel;b</r>")).text, "a&#x7;b");
}

#[test]
fn line_ends_follow_the_declared_version() {
    // XML 1.1 folds NEL and LINE SEPARATOR into `#xA`; XML 1.0 leaves both.
    assert_eq!(
        collect(&format!("{DECL_1_1}<r>a\u{85}b\u{2028}c\r\nd</r>")).text,
        "a\nb\nc\nd"
    );
    assert_eq!(
        collect("<r>a\u{85}b\u{2028}c\r\nd</r>").text,
        "a\u{85}b\u{2028}c\nd"
    );
}
