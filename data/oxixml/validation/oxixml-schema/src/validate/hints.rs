//! Instance-driven schema loading: the `xsi:schemaLocation` and
//! `xsi:noNamespaceSchemaLocation` hints of
//! [§4.3.2](https://www.w3.org/TR/xmlschema-1/#xsi_schemaLocation).
//!
//! An instance may name the schema documents it expects to be assessed
//! against. The hint is exactly that — a hint: the components already in hand
//! take precedence, a document that cannot be retrieved is not by itself an
//! error, and a processor is free to ignore the whole mechanism. This module
//! implements the retrieval half; the precedence half lives in the assessment
//! itself, which consults the assembled schema first and only falls back to
//! what the hints produced.
//!
//! Retrieval is **file-system only**. A hint whose location carries a scheme —
//! `http:`, `ftp:`, anything with a `//` authority — is ignored rather than
//! fetched, so an instance can never make assessment reach the network.

use std::collections::HashSet;

use oxixml_dom::{Document, NodeKind};

use crate::compile::SchemaSet;
use crate::names::{XSI_NS, XsdVersion};
use crate::resolver::join_location;
use crate::schema::Schema;

/// How many distinct schema documents one instance may name before the rest of
/// its hints are ignored.
///
/// Every hint costs a file-system lookup and, when it resolves, a parsed
/// document held for the length of the assessment, so an instance naming an
/// unbounded number of them would turn a single validation into an unbounded
/// amount of work. Real instances name one or two — the whole W3C XML Schema
/// Test Suite never names more than two — so the ceiling is far out of the way
/// of anything written to be assessed.
const MAX_HINTED_DOCUMENTS: usize = 256;

/// Assembles the schema documents `document` names, resolved against `base`.
///
/// Returns [`None`] when the instance names nothing, when nothing it names can
/// be retrieved, or when what was retrieved does not assemble: none of those is
/// an error here, because §4.3.2 leaves the whole mechanism to the processor's
/// discretion. What *is* an error — a reference the resulting schema cannot
/// satisfy, say — is reported later, against the instance, by the assessment
/// that consults it.
pub(crate) fn load(version: XsdVersion, document: &Document, base: &str) -> Option<Schema> {
    let locations = locations(document);
    if locations.is_empty() {
        return None;
    }
    let mut set = SchemaSet::new().with_version(version);
    for location in locations {
        // A hint that names a document which is missing, unreadable or not a
        // schema document contributes nothing and stops nothing: the remaining
        // hints are still followed.
        let _ = set.add_file(join_location(&location, Some(base)));
    }
    if set.is_empty() {
        return None;
    }
    set.compile().ok()
}

/// Every schema-document location the instance names, in document order and
/// without repeats.
///
/// `xsi:schemaLocation` is a whitespace-separated list of *pairs*, each a
/// namespace name followed by a location; a trailing token with no location
/// after it names nothing and is dropped. `xsi:noNamespaceSchemaLocation` is a
/// single location. Both may appear on any element, not only the document
/// element, so the whole tree is walked — with an explicit stack, so that a
/// deep instance cannot exhaust the stack here before assessment has even
/// begun.
fn locations(document: &Document) -> Vec<String> {
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    let Some(root) = document.document_element() else {
        return found;
    };
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if let Some(text) = document.get_attribute_ns(node, Some(XSI_NS), "schemaLocation") {
            let mut tokens = text.split_whitespace();
            // `next()` twice per turn: the namespace name is skipped, since
            // what the location declares is checked when it is assembled.
            while let (Some(_), Some(location)) = (tokens.next(), tokens.next()) {
                push(&mut found, &mut seen, location);
            }
        }
        if let Some(text) =
            document.get_attribute_ns(node, Some(XSI_NS), "noNamespaceSchemaLocation")
        {
            for location in text.split_whitespace() {
                push(&mut found, &mut seen, location);
            }
        }
        // Pushed in reverse so that popping takes the children in document
        // order: two elements may hint different locations for one namespace,
        // and which of them is read first decides which components the
        // assembled hints end up holding.
        let children: Vec<_> = document
            .children(node)
            .filter(|child| document.kind(*child) == Some(NodeKind::Element))
            .collect();
        stack.extend(children.into_iter().rev());
    }
    found
}

/// Records one location, unless it repeats an earlier one, names a scheme that
/// is not the file system, or would take the list past its ceiling.
fn push(found: &mut Vec<String>, seen: &mut HashSet<String>, location: &str) {
    if found.len() >= MAX_HINTED_DOCUMENTS || !is_retrievable(location) {
        return;
    }
    if seen.insert(location.to_owned()) {
        found.push(location.to_owned());
    }
}

/// Whether a location names something this crate is willing to retrieve.
///
/// A relative reference and a `file:` URI are read from the file system;
/// anything else with an authority — `http://…`, `ftp://…` — is left alone.
fn is_retrievable(location: &str) -> bool {
    match location.split_once("://") {
        Some((scheme, _)) => scheme.eq_ignore_ascii_case("file"),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Document {
        oxixml_dom::build::parse_str(text).expect("a well-formed instance")
    }

    #[test]
    fn pairs_contribute_their_second_token() {
        let document = parse(
            r#"<root xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                     xsi:schemaLocation="urn:a a.xsd urn:b b.xsd"/>"#,
        );
        assert_eq!(locations(&document), vec!["a.xsd", "b.xsd"]);
    }

    #[test]
    fn a_trailing_namespace_without_a_location_is_dropped() {
        let document = parse(
            r#"<root xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                     xsi:schemaLocation="urn:a a.xsd urn:b"/>"#,
        );
        assert_eq!(locations(&document), vec!["a.xsd"]);
    }

    #[test]
    fn hints_are_collected_from_the_whole_tree_in_document_order() {
        let document = parse(
            r#"<root xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                     xsi:noNamespaceSchemaLocation="a.xsd">
                 <first xsi:schemaLocation="urn:b b.xsd urn:c a.xsd"/>
                 <second xsi:schemaLocation="urn:d d.xsd">
                   <deep xsi:noNamespaceSchemaLocation="e.xsd"/>
                 </second>
                 <third xsi:schemaLocation="urn:f f.xsd"/>
               </root>"#,
        );
        assert_eq!(
            locations(&document),
            vec!["a.xsd", "b.xsd", "d.xsd", "e.xsd", "f.xsd"]
        );
    }

    #[test]
    fn a_remote_scheme_is_never_retrieved() {
        assert!(is_retrievable("b.xsd"));
        assert!(is_retrievable("../shared/b.xsd"));
        assert!(is_retrievable("file:///tmp/b.xsd"));
        assert!(!is_retrievable("http://www.w3.org/2001/xml.xsd"));
        assert!(!is_retrievable("HTTPS://example.invalid/b.xsd"));
    }
}
