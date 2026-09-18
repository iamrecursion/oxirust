//! Helpers shared by the integration tests.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use oxixml_dom::{Document, NodeId, build};

/// The HMAC secret the W3C XML Signature 1.1 interoperability documents from
/// Oracle are keyed with.
///
/// The documents themselves carry no key — an HMAC secret cannot be published
/// inside the signature it authenticates — so it is recovered here as a
/// constant. It is the only piece of the suite that is not self-describing.
pub const INTEROP_HMAC_SECRET: &[u8] = b"testkey";

/// The `xmldsig-core1-interop` directory, or [`None`] when the suite is not
/// configured.
pub fn interop_directory() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("OXIXML_XML_SUITES_DIR")?);
    let directory = root.join("xmldsig-core1-interop");
    directory.is_dir().then_some(directory)
}

/// Every `*.xml` file below `directory`, sorted.
pub fn xml_files(directory: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "xml") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Decodes a hexadecimal string, ignoring ASCII whitespace.
///
/// The published key and signature vectors these tests use are all printed in
/// hexadecimal, so they are transcribed that way rather than converted, which
/// would make them impossible to check against their source.
pub fn hex(text: &str) -> Vec<u8> {
    let digits: Vec<u32> = text
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .map(|character| {
            character
                .to_digit(16)
                .expect("the transcribed vector is hexadecimal")
        })
        .collect();
    assert!(
        digits.len() % 2 == 0,
        "a hexadecimal vector has an even number of digits"
    );
    digits
        .chunks(2)
        .map(|pair| {
            let high = pair.first().copied().unwrap_or(0);
            let low = pair.get(1).copied().unwrap_or(0);
            u8::try_from((high << 4) | low).unwrap_or(0)
        })
        .collect()
}

/// Parses a document, panicking with the parser's message on failure.
pub fn parse(xml: &str) -> Document {
    build::parse_str(xml).expect("the test input is well-formed XML")
}

/// Parses a document whose element nesting may exceed the reader's default
/// depth limit, raising that limit to `depth`.
pub fn parse_nested(xml: &str, depth: usize) -> Document {
    let options = build::BuilderOptions::new().with_max_element_depth(depth);
    build::parse_str_with(xml, &options).expect("the test input is well-formed XML")
}

/// The single element with the given qualified name.
pub fn element_named(document: &Document, name: &str) -> NodeId {
    document
        .get_elements_by_tag_name(document.root(), name)
        .next()
        .unwrap_or_else(|| panic!("expected a <{name}> element"))
}

/// Flips the last character of an element's text content, which is the cheapest
/// way to make a signed payload no longer digest to its `DigestValue`.
pub fn tamper_text(document: &mut Document, element: NodeId) {
    let original = document.text_content(element).unwrap_or_default();
    let mut tampered = original.clone();
    tampered.push('!');
    document
        .set_text_content(element, tampered)
        .expect("text content can be replaced");
}
