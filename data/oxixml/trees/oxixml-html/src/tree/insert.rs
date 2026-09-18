//! Node insertion: the "appropriate place for inserting a node" algorithm with
//! foster parenting, and the element/text/comment insertion steps built on it
//! ([HTML § 13.2.6.1](https://html.spec.whatwg.org/multipage/parsing.html#creating-and-inserting-nodes)).

use oxixml_dom::NodeId;

use super::dom;
use super::{OpenElement, TreeBuilder};
use crate::error::codes;
use crate::ns::Namespace;
use crate::token::{Attribute, ProcessingInstruction, Tag, Token};

/// Where a node is about to be inserted: a parent and an optional child to
/// insert before.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InsertionPoint {
    pub(crate) parent: NodeId,
    pub(crate) before: Option<NodeId>,
}

/// One attribute after the foreign-content adjustments have been applied.
pub(crate) struct AdjustedAttribute {
    /// The qualified name to give the attribute.
    pub(crate) qualified_name: String,
    /// The namespace to put it in, or `None` for no namespace.
    pub(crate) namespace: Option<&'static str>,
    /// The value.
    pub(crate) value: String,
}

impl TreeBuilder<'_> {
    /// Maps a `template` element to the document fragment that holds its
    /// contents, and every other node to itself.
    pub(crate) fn content_of(&self, id: NodeId) -> NodeId {
        self.template_contents
            .iter()
            .find(|(template, _)| *template == id)
            .map_or(id, |(_, content)| *content)
    }

    /// The "appropriate place for inserting a node" algorithm.
    pub(crate) fn appropriate_place(&mut self, override_target: Option<NodeId>) -> InsertionPoint {
        let target = override_target.or_else(|| self.current_node().map(|node| node.id));
        let Some(target) = target else {
            return InsertionPoint {
                parent: self.fragment_root.unwrap_or_else(|| self.document.root()),
                before: None,
            };
        };

        let target_is_table_ish = self
            .index_of(target)
            .and_then(|index| self.open.get(index))
            .is_some_and(|node| node.is_html_any(&["table", "tbody", "tfoot", "thead", "tr"]));

        let point = if self.foster_parenting && target_is_table_ish {
            self.foster_parent_location()
        } else {
            InsertionPoint {
                parent: target,
                before: None,
            }
        };

        // "If the adjusted insertion location is inside a template element,
        // let it instead be inside the template element's template contents."
        let content = self.content_of(point.parent);
        if content == point.parent {
            point
        } else {
            InsertionPoint {
                parent: content,
                before: None,
            }
        }
    }

    /// Steps 2.1–2.5 of the "appropriate place" algorithm: where content is
    /// foster parented to when it would otherwise land inside a table.
    fn foster_parent_location(&mut self) -> InsertionPoint {
        let last_template = self.open.iter().rposition(|node| node.is_html("template"));
        let last_table = self.open.iter().rposition(|node| node.is_html("table"));

        if let Some(template_index) = last_template {
            if last_table.is_none_or(|table_index| template_index > table_index) {
                let template = self.open[template_index].id;
                return InsertionPoint {
                    parent: self.content_of(template),
                    before: None,
                };
            }
        }

        let Some(table_index) = last_table else {
            let parent = self
                .open
                .first()
                .map_or_else(|| self.document.root(), |node| node.id);
            return InsertionPoint {
                parent,
                before: None,
            };
        };

        let table = self.open[table_index].id;
        if let Some(parent) = self.document.parent(table) {
            return InsertionPoint {
                parent,
                before: Some(table),
            };
        }
        let previous = self
            .open
            .get(table_index.wrapping_sub(1))
            .map_or(table, |node| node.id);
        InsertionPoint {
            parent: previous,
            before: None,
        }
    }

    /// Inserts an already-created node at an insertion point.
    pub(crate) fn insert_at(&mut self, point: InsertionPoint, node: NodeId) {
        let result = self
            .document
            .insert_before(point.parent, node, point.before);
        if result.is_err() {
            // The only way this can fail is a hierarchy constraint of the
            // document node (for instance a second document element), in which
            // case the standard's own "ignore the token" recovery applies.
            self.error(codes::UNEXPECTED_START_TAG);
        }
    }

    // ------------------------------------------------------------- elements

    /// "Create an element for a token".
    pub(crate) fn create_element_for_token(
        &mut self,
        tag: &Tag,
        namespace: Namespace,
        attributes: Vec<AdjustedAttribute>,
    ) -> NodeId {
        let name = dom::element_name(&tag.name);
        let rewritten = matches!(name, std::borrow::Cow::Owned(_));
        if rewritten {
            self.error(codes::UNREPRESENTABLE_NAME);
        }
        let element = match self
            .document
            .create_element_ns(Some(namespace.uri()), name.as_ref())
        {
            Ok(element) => element,
            Err(_) => {
                self.error(codes::UNREPRESENTABLE_NAME);
                self.document
                    .create_element_ns(Some(namespace.uri()), "_")
                    .unwrap_or_else(|_| self.document.create_document_fragment())
            }
        };
        if rewritten {
            self.record_original(element, &tag.name);
        }
        for attribute in attributes {
            self.set_adjusted_attribute(element, &attribute);
        }
        element
    }

    fn set_adjusted_attribute(&mut self, element: NodeId, attribute: &AdjustedAttribute) {
        let value = dom::character_data(&attribute.value);
        if matches!(value, std::borrow::Cow::Owned(_)) {
            self.error(codes::UNREPRESENTABLE_CHARACTER_DATA);
        }
        let result = match attribute.namespace {
            Some(namespace) => self
                .document
                .set_attribute_ns(
                    element,
                    Some(namespace),
                    &attribute.qualified_name,
                    value.into_owned(),
                )
                .map(|_| ()),
            None => {
                let name = dom::attribute_name(&attribute.qualified_name);
                let rewritten = matches!(name, std::borrow::Cow::Owned(_));
                if rewritten {
                    self.error(codes::UNREPRESENTABLE_NAME);
                }
                if self.document.has_attribute(element, name.as_ref()) {
                    // Duplicates keep the first value, as the tokenizer does.
                    return;
                }
                match self
                    .document
                    .set_attribute(element, name.as_ref(), value.into_owned())
                {
                    Ok(node) => {
                        if rewritten {
                            self.record_original(node, &attribute.qualified_name);
                        }
                        Ok(())
                    }
                    Err(error) => Err(error),
                }
            }
        };
        if result.is_err() {
            self.error(codes::UNREPRESENTABLE_NAME);
        }
    }

    /// Remembers the HTML spelling of a name or of character data that had to
    /// be rewritten to fit an XML DOM.
    pub(crate) fn record_original(&mut self, node: NodeId, original: &str) {
        self.originals.push((node, original.to_owned()));
    }

    /// "Insert an HTML element for a token".
    pub(crate) fn insert_html_element(&mut self, tag: &Tag) -> NodeId {
        self.insert_element(tag, Namespace::Html, plain_attributes(&tag.attributes))
    }

    /// Inserts an element in any namespace and pushes it onto the stack of open
    /// elements, unless [`ParseOptions::max_depth`](crate::ParseOptions::max_depth)
    /// has already been reached.
    ///
    /// This is the single point every element insertion funnels through
    /// (directly, or via [`TreeBuilder::insert_html_element`]), so it is
    /// where the depth ceiling is enforced: every tree construction algorithm
    /// that walks [`TreeBuilder::open`] from the top down (matching an end
    /// tag, the six element scopes, the adoption agency algorithm) is thereby
    /// bounded by the same ceiling, without needing a check of its own. The
    /// element is still created and inserted into the document either way —
    /// only whether it becomes *tracked as open* depends on the depth check —
    /// so content past the ceiling is not lost, it is simply no longer
    /// distinguished from a sibling of whatever was current when the ceiling
    /// was first reached.
    pub(crate) fn insert_element(
        &mut self,
        tag: &Tag,
        namespace: Namespace,
        attributes: Vec<AdjustedAttribute>,
    ) -> NodeId {
        let element = self.create_element_for_token(tag, namespace, attributes);
        let point = self.appropriate_place(None);
        self.insert_at(point, element);
        if namespace == Namespace::Html && tag.name == "template" {
            let content = self.document.create_document_fragment();
            self.template_contents.push((element, content));
        }
        if self.open.len() < self.max_depth {
            self.open
                .push(OpenElement::new(element, namespace, tag.name.clone()));
        } else if !self.max_depth_exceeded {
            self.max_depth_exceeded = true;
            self.error(codes::MAX_DEPTH_EXCEEDED);
        }
        element
    }

    /// Creates an element for a token without inserting it anywhere, which the
    /// adoption agency algorithm and "reconstruct the active formatting
    /// elements" both need.
    pub(crate) fn create_detached_html_element(&mut self, tag: &Tag) -> NodeId {
        self.create_element_for_token(tag, Namespace::Html, plain_attributes(&tag.attributes))
    }

    // ------------------------------------------------------------ character data

    /// "Insert a character": text goes into the text node immediately before
    /// the insertion position when there is one, and into a new text node
    /// otherwise.
    pub(crate) fn insert_characters(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let point = self.appropriate_place(None);
        if point.parent == self.document.root() {
            // "If the adjusted insertion location is in a Document node, then
            // return." Text is never a child of a document.
            return;
        }
        let clean = dom::character_data(text);
        if matches!(clean, std::borrow::Cow::Owned(_)) {
            self.error(codes::UNREPRESENTABLE_CHARACTER_DATA);
        }
        if clean.is_empty() {
            return;
        }
        let previous = match point.before {
            Some(reference) => self.document.previous_sibling(reference),
            None => self.document.last_child(point.parent),
        };
        if let Some(previous) = previous {
            if self.document.kind(previous) == Some(oxixml_dom::NodeKind::Text) {
                let mut existing = self
                    .document
                    .node_value(previous)
                    .unwrap_or_default()
                    .to_owned();
                existing.push_str(&clean);
                let result = self.document.set_node_value(previous, existing);
                debug_assert!(
                    result.is_ok(),
                    "`previous` was just confirmed to be a Text node above, which always \
                     accepts a node value, and `existing` is that node's own already-XML-safe \
                     value with `clean` appended -- `clean` already passed through \
                     dom::character_data, so the concatenation is still made only of characters \
                     the XML Char production allows: {result:?}"
                );
                return;
            }
        }
        let Ok(node) = self.document.create_text_node(clean.into_owned()) else {
            self.error(codes::UNREPRESENTABLE_CHARACTER_DATA);
            return;
        };
        self.insert_at(point, node);
    }

    /// "Insert a comment", optionally at an explicit position.
    pub(crate) fn insert_comment(&mut self, data: &str, position: Option<InsertionPoint>) {
        let clean = dom::comment_data(data);
        let rewritten = matches!(clean, std::borrow::Cow::Owned(_));
        if rewritten {
            self.error(codes::UNREPRESENTABLE_CHARACTER_DATA);
        }
        let Ok(node) = self.document.create_comment(clean.into_owned()) else {
            self.error(codes::UNREPRESENTABLE_CHARACTER_DATA);
            return;
        };
        if rewritten {
            self.record_original(node, data);
        }
        let point = match position {
            Some(point) => point,
            None => self.appropriate_place(None),
        };
        self.insert_at(point, node);
    }

    /// "Insert a processing instruction", optionally at an explicit position.
    pub(crate) fn insert_processing_instruction(
        &mut self,
        instruction: &ProcessingInstruction,
        position: Option<InsertionPoint>,
    ) {
        let data = dom::character_data(&instruction.data);
        if matches!(data, std::borrow::Cow::Owned(_)) {
            self.error(codes::UNREPRESENTABLE_CHARACTER_DATA);
        }
        let Ok(node) = self
            .document
            .create_processing_instruction(&instruction.target, data.into_owned())
        else {
            self.error(codes::UNREPRESENTABLE_NAME);
            return;
        };
        let point = match position {
            Some(point) => point,
            None => self.appropriate_place(None),
        };
        self.insert_at(point, node);
    }

    /// Inserts a comment or processing instruction token.
    pub(crate) fn insert_markup(&mut self, token: &Token, position: Option<InsertionPoint>) {
        match token {
            Token::Comment(data) => self.insert_comment(data, position),
            Token::ProcessingInstruction(instruction) => {
                self.insert_processing_instruction(instruction, position);
            }
            _ => {}
        }
    }

    /// The insertion point "at the end of the Document", used by the modes that
    /// run after the document element has been closed.
    pub(crate) fn document_end(&self) -> InsertionPoint {
        InsertionPoint {
            parent: self.document.root(),
            before: None,
        }
    }

    /// The insertion point at the end of the `html` element, used by the "after
    /// body" insertion mode.
    pub(crate) fn html_element_end(&self) -> InsertionPoint {
        InsertionPoint {
            parent: self
                .open
                .first()
                .map_or_else(|| self.document.root(), |node| node.id),
            before: None,
        }
    }
}

/// Turns tokenizer attributes into DOM attributes with no namespace, which is
/// what every HTML element gets.
pub(crate) fn plain_attributes(attributes: &[Attribute]) -> Vec<AdjustedAttribute> {
    attributes
        .iter()
        .map(|attribute| AdjustedAttribute {
            qualified_name: attribute.name.clone(),
            namespace: None,
            value: attribute.value.clone(),
        })
        .collect()
}
