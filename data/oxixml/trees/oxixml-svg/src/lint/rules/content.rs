//! Content-model checks: [`check_content_model`] validates one element's
//! *direct* children against [`crate::names::content::allows_child`] and
//! [`crate::names::content::allows_character_data`].

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::lint::diagnostic::{Diagnostic, codes};
use crate::names::content::{allows_character_data, allows_child};
use crate::names::elements::ElementId;
use crate::ns::SVG_NAMESPACE;

/// Checks every direct child of `parent` (a known-kind SVG element)
/// against its content model: an SVG-namespace child element
/// [`allows_child`] rejects is [`codes::CONTENT_MODEL_VIOLATION`]
/// (attached to the *child*, not `parent`); non-whitespace character data
/// where [`allows_character_data`] is `false` is
/// [`codes::CHARACTER_DATA_NOT_ALLOWED`] (attached to the text node).
///
/// A child in a foreign namespace (or an unknown SVG-namespace element,
/// which [`super::Walker::walk`] already reports under
/// [`codes::UNKNOWN_ELEMENT`]) cannot be checked against a content model
/// this crate's [`ElementId`] catalog does not have an entry for, and is
/// silently skipped here. Comment and processing-instruction children are
/// never checked: a comment is legal everywhere regardless of content
/// model.
pub(super) fn check_content_model(
    doc: &Document,
    parent: NodeId,
    parent_kind: ElementId,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for child in doc.children(parent) {
        match doc.kind(child) {
            Some(NodeKind::Element) => {
                check_child_element(doc, parent_kind, child, diagnostics);
            }
            Some(NodeKind::Text | NodeKind::CdataSection) => {
                check_character_data(doc, parent_kind, child, diagnostics);
            }
            _ => {}
        }
    }
}

fn check_child_element(
    doc: &Document,
    parent_kind: ElementId,
    child: NodeId,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(element) = doc.element(child) else {
        return;
    };
    if element.namespace_uri() != Some(SVG_NAMESPACE) {
        return;
    }
    let Some(child_kind) = ElementId::from_name(element.local_name()) else {
        return;
    };
    if !allows_child(parent_kind, child_kind) {
        diagnostics.push(Diagnostic::warning(
            codes::CONTENT_MODEL_VIOLATION,
            format!(
                "`<{}>` is not a valid child of `<{}>`",
                child_kind.as_str(),
                parent_kind.as_str()
            ),
            child,
        ));
    }
}

fn check_character_data(
    doc: &Document,
    parent_kind: ElementId,
    child: NodeId,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if allows_character_data(parent_kind) {
        return;
    }
    let text = doc.character_data(child).unwrap_or_default();
    if text.trim().is_empty() {
        return;
    }
    diagnostics.push(Diagnostic::warning(
        codes::CHARACTER_DATA_NOT_ALLOWED,
        format!(
            "character data is not allowed inside `<{}>`",
            parent_kind.as_str()
        ),
        child,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build::parse_str;

    fn children_of_root(doc: &Document) -> NodeId {
        doc.document_element().expect("document element")
    }

    #[test]
    fn rect_in_rect_is_a_content_model_violation() {
        let doc =
            parse_str(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect><rect/></rect></svg>"##)
                .expect("well-formed");
        let root = children_of_root(&doc);
        let rect = doc.first_child(root).expect("outer rect");
        let mut diagnostics = Vec::new();
        check_content_model(&doc, rect, ElementId::Rect, &mut diagnostics);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code(), codes::CONTENT_MODEL_VIOLATION);
    }

    #[test]
    fn rect_in_g_does_not_fire() {
        let doc = parse_str(r##"<svg xmlns="http://www.w3.org/2000/svg"><g><rect/></g></svg>"##)
            .expect("well-formed");
        let root = children_of_root(&doc);
        let g = doc.first_child(root).expect("g");
        let mut diagnostics = Vec::new();
        check_content_model(&doc, g, ElementId::G, &mut diagnostics);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn whitespace_only_text_in_g_is_fine() {
        let doc = parse_str("<svg xmlns=\"http://www.w3.org/2000/svg\"><g>\n  </g></svg>")
            .expect("well-formed");
        let root = children_of_root(&doc);
        let g = doc.first_child(root).expect("g");
        let mut diagnostics = Vec::new();
        check_content_model(&doc, g, ElementId::G, &mut diagnostics);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn real_text_in_g_fires_character_data_not_allowed() {
        let doc = parse_str(r##"<svg xmlns="http://www.w3.org/2000/svg"><g>hello</g></svg>"##)
            .expect("well-formed");
        let root = children_of_root(&doc);
        let g = doc.first_child(root).expect("g");
        let mut diagnostics = Vec::new();
        check_content_model(&doc, g, ElementId::G, &mut diagnostics);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code(), codes::CHARACTER_DATA_NOT_ALLOWED);
        assert_eq!(
            diagnostics[0].node(),
            doc.first_child(g).expect("text node")
        );
    }

    #[test]
    fn text_in_text_element_does_not_fire() {
        let doc =
            parse_str(r##"<svg xmlns="http://www.w3.org/2000/svg"><text>hello</text></svg>"##)
                .expect("well-formed");
        let root = children_of_root(&doc);
        let text_el = doc.first_child(root).expect("text element");
        let mut diagnostics = Vec::new();
        check_content_model(&doc, text_el, ElementId::Text, &mut diagnostics);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn foreign_namespace_children_are_not_checked() {
        let doc = parse_str(
            r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:x="urn:example"><g><x:weird/></g></svg>"##,
        )
        .expect("well-formed");
        let root = children_of_root(&doc);
        let g = doc.first_child(root).expect("g");
        let mut diagnostics = Vec::new();
        check_content_model(&doc, g, ElementId::G, &mut diagnostics);
        assert!(diagnostics.is_empty());
    }
}
