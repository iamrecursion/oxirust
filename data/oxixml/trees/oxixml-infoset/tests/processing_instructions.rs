//! What a processing instruction's `[content]` property holds.
//!
//! [XML Infoset §2.6](https://www.w3.org/TR/xml-infoset/#infoitem.pi) defines it as "a string
//! representing the content of the processing instruction, excluding the target and any white
//! space immediately following it" — the whole `S` run of
//! [XML 1.0 §2.6](https://www.w3.org/TR/xml/#sec-pi), not just its first character. XDM 3.1 §6.5
//! makes that string the node's string value, so `<?pro  my document?>` and `<?pro my document?>`
//! both hold `my document`, and trailing white space is content and stays.

use oxixml_infoset::{NodeKind, Tree, XdmError, XdmNode};

/// The string value of the one processing instruction inside `<a>`.
fn content(xml: &str) -> Result<String, XdmError> {
    let tree = Tree::parse_str(xml)?;
    let instruction = tree
        .document_element()
        .expect("a document element")
        .children()
        .first()
        .cloned()
        .expect("one child");
    assert_eq!(instruction.node_kind(), NodeKind::ProcessingInstruction);
    Ok(instruction.string_value())
}

#[test]
fn the_whole_separating_whitespace_run_is_excluded() -> Result<(), XdmError> {
    assert_eq!(content("<a><?pro my document?></a>")?, "my document");
    assert_eq!(content("<a><?pro  my document?></a>")?, "my document");
    assert_eq!(content("<a><?pro     my document?></a>")?, "my document");
    Ok(())
}

#[test]
fn every_xml_whitespace_character_separates() -> Result<(), XdmError> {
    assert_eq!(content("<a><?pro\tdata?></a>")?, "data");
    assert_eq!(content("<a><?pro\ndata?></a>")?, "data");
    assert_eq!(content("<a><?pro\r\ndata?></a>")?, "data");
    assert_eq!(content("<a><?pro \t\r\n data?></a>")?, "data");
    Ok(())
}

#[test]
fn whitespace_inside_and_after_the_content_is_kept() -> Result<(), XdmError> {
    // Only the run that separates the content from the target is excluded.
    assert_eq!(content("<a><?pro a  b?></a>")?, "a  b");
    assert_eq!(content("<a><?pro a  ?></a>")?, "a  ");
    Ok(())
}

#[test]
fn a_target_with_no_content_has_an_empty_string_value() -> Result<(), XdmError> {
    assert_eq!(content("<a><?pro?></a>")?, "");
    assert_eq!(content("<a><?pro ?></a>")?, "");
    assert_eq!(content("<a><?pro   ?></a>")?, "");
    Ok(())
}
