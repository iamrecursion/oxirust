//! Driving the derivative algorithm over an instance document.
//!
//! The validator is written as a push interface — `start_element`, `characters`,
//! `end_element` — so that any event source can feed it. Two drivers are
//! provided: one over an [`oxixml_dom::Document`] and one that parses a string
//! into such a document first.
//!
//! # Error recovery
//!
//! A pure derivative validator answers yes or no. To report *several* problems,
//! this one recovers: an element the schema does not allow is reported and its
//! whole subtree skipped, a rejected attribute is reported and dropped, and an
//! element whose content is incomplete resumes with the continuation its `after`
//! pattern would have produced. Recovery only ever affects which *additional*
//! errors are reported — a document with no reported error is valid.

use std::rc::Rc;

use oxixml_dom::{Document, NodeId, NodeKind, XMLNS_NAMESPACE};

use crate::compile::Schema;
use crate::datatype::DatatypeContext;
use crate::deriv::Engine;
use crate::error::{RelaxNgError, ValidationErrorKind};
use crate::pattern::{NOT_ALLOWED, PatId};
use crate::tree::{NamespaceMap, is_whitespace};

/// One attribute of an element, as the data model sees it.
#[derive(Debug, Clone)]
pub(crate) struct Attribute<'a> {
    /// The namespace name, or `""` when the attribute is in no namespace.
    pub namespace: &'a str,
    /// The local name.
    pub local: &'a str,
    /// The normalized value.
    pub value: &'a str,
}

struct Frame {
    name: String,
    continuation: PatId,
    text: String,
    saw_element: bool,
    context: DatatypeContext,
}

/// The push-driven validator.
pub(crate) struct Validator<'a> {
    engine: Engine<'a>,
    current: PatId,
    frames: Vec<Frame>,
    errors: Vec<RelaxNgError>,
    skip: usize,
}

impl<'a> Validator<'a> {
    /// Starts a validation run against `schema`.
    pub(crate) fn new(schema: &'a Schema) -> Self {
        let engine = Engine::new(schema);
        let current = engine.start();
        Self {
            engine,
            current,
            frames: Vec::new(),
            errors: Vec::new(),
            skip: 0,
        }
    }

    fn path(&self) -> String {
        if self.frames.is_empty() {
            return "/".to_owned();
        }
        let mut path = String::new();
        for frame in &self.frames {
            path.push('/');
            path.push_str(&frame.name);
        }
        path
    }

    fn report(&mut self, kind: ValidationErrorKind, message: String) {
        self.errors.push(RelaxNgError::new(kind, message));
    }

    /// Feeds an element start tag together with all of its attributes.
    pub(crate) fn start_element(
        &mut self,
        namespace: &str,
        local: &str,
        attributes: &[Attribute<'_>],
        namespaces: Rc<NamespaceMap>,
    ) {
        if self.skip > 0 {
            self.skip += 1;
            return;
        }
        self.flush_text(true);

        let display = display_name(namespace, local);
        let opened = self
            .engine
            .start_tag_open_deriv(self.current, namespace, local);
        if opened == NOT_ALLOWED {
            let expected = self.engine.expected(self.current);
            let path = self.path();
            let message = if expected.is_empty() {
                format!("element `{display}` is not allowed here (in {path})")
            } else {
                format!(
                    "element `{display}` is not allowed here (in {path}); expected one of: {}",
                    expected.join(", ")
                )
            };
            self.report(ValidationErrorKind::UnexpectedElement, message);
            self.skip = 1;
            return;
        }

        let context = DatatypeContext {
            namespaces: Rc::clone(&namespaces),
        };
        let mut pattern = opened;
        for attribute in attributes {
            let derived = self.engine.att_deriv(
                pattern,
                attribute.namespace,
                attribute.local,
                attribute.value,
                &context,
            );
            if derived == NOT_ALLOWED {
                let name = display_name(attribute.namespace, attribute.local);
                let message = format!(
                    "attribute `{name}` is not allowed on element `{display}`, or its value \
                     `{}` does not match the schema",
                    attribute.value
                );
                self.report(ValidationErrorKind::UnexpectedAttribute, message);
            } else {
                pattern = derived;
            }
        }

        let closed = self.engine.start_tag_close_deriv(pattern);
        if closed == NOT_ALLOWED {
            let expected = self.engine.expected(pattern);
            let message = if expected.is_empty() {
                format!("element `{display}` is missing a required attribute")
            } else {
                format!(
                    "element `{display}` is missing a required attribute; expected one of: {}",
                    expected.join(", ")
                )
            };
            self.report(ValidationErrorKind::UnexpectedAttribute, message);
        } else {
            pattern = closed;
        }

        let continuation = self.engine.recovery_continuation(pattern);
        self.frames.push(Frame {
            name: display,
            continuation,
            text: String::new(),
            saw_element: false,
            context,
        });
        self.current = pattern;
    }

    /// Feeds character data. Consecutive calls accumulate into a single string,
    /// which is what the data model requires: comments and processing
    /// instructions do not split character data.
    pub(crate) fn characters(&mut self, text: &str) {
        if self.skip > 0 {
            return;
        }
        if let Some(frame) = self.frames.last_mut() {
            frame.text.push_str(text);
        }
    }

    /// Feeds an element end tag.
    pub(crate) fn end_element(&mut self) {
        if self.skip > 0 {
            self.skip -= 1;
            return;
        }
        self.flush_text(false);
        let Some(frame) = self.frames.pop() else {
            return;
        };
        let ended = self.engine.end_tag_deriv(self.current);
        if ended == NOT_ALLOWED {
            let expected = self.engine.expected(self.current);
            let message = if expected.is_empty() {
                format!("the content of element `{}` is incomplete", frame.name)
            } else {
                format!(
                    "the content of element `{}` is incomplete; expected one of: {}",
                    frame.name,
                    expected.join(", ")
                )
            };
            self.report(ValidationErrorKind::IncompleteContent, message);
            self.current = frame.continuation;
        } else {
            self.current = ended;
        }
    }

    /// Applies the accumulated character data of the innermost frame.
    ///
    /// `before_element` distinguishes the two cases of `childrenDeriv`: a lone
    /// text child is matched as-is (and whitespace is optional), whereas in mixed
    /// content whitespace-only strings are stripped.
    fn flush_text(&mut self, before_element: bool) {
        let (text, context, name, only_child) = {
            let Some(frame) = self.frames.last_mut() else {
                return;
            };
            let text = std::mem::take(&mut frame.text);
            let only_child = !frame.saw_element && !before_element;
            if before_element {
                frame.saw_element = true;
            }
            (text, frame.context.clone(), frame.name.clone(), only_child)
        };

        if only_child {
            let derived = self.engine.text_deriv(self.current, &text, &context);
            if is_whitespace(&text) {
                self.current = self.engine.choice(self.current, derived);
            } else if derived == NOT_ALLOWED {
                self.report(
                    ValidationErrorKind::UnexpectedText,
                    format!(
                        "character data is not allowed in element `{name}`, or `{}` does not \
                         match the schema",
                        summarize(&text)
                    ),
                );
            } else {
                self.current = derived;
            }
            return;
        }

        if text.is_empty() || is_whitespace(&text) {
            return;
        }
        let derived = self.engine.text_deriv(self.current, &text, &context);
        if derived == NOT_ALLOWED {
            self.report(
                ValidationErrorKind::UnexpectedText,
                format!(
                    "character data is not allowed here in element `{name}`, or `{}` does not \
                     match the schema",
                    summarize(&text)
                ),
            );
        } else {
            self.current = derived;
        }
    }

    /// Finishes the run, returning every error found.
    pub(crate) fn finish(mut self) -> Vec<RelaxNgError> {
        if self.errors.is_empty() && !self.engine.nullable(self.current) {
            self.errors.push(RelaxNgError::new(
                ValidationErrorKind::IncompleteContent,
                "the document does not match the schema".to_owned(),
            ));
        }
        self.errors
    }
}

fn display_name(namespace: &str, local: &str) -> String {
    if namespace.is_empty() {
        local.to_owned()
    } else {
        format!("{{{namespace}}}{local}")
    }
}

fn summarize(text: &str) -> String {
    let trimmed = crate::tree::trim_whitespace(text);
    if trimmed.chars().count() <= 40 {
        return trimmed.to_owned();
    }
    let head: String = trimmed.chars().take(37).collect();
    format!("{head}...")
}

/// Validates a whole DOM document.
pub(crate) fn validate_document(schema: &Schema, document: &Document) -> Vec<RelaxNgError> {
    let mut validator = Validator::new(schema);
    let Some(root) = document.document_element() else {
        return vec![RelaxNgError::new(
            ValidationErrorKind::NotWellFormed,
            "the instance has no document element".to_owned(),
        )];
    };
    let namespaces = Rc::new(NamespaceMap::initial());
    walk(&mut validator, document, root, &namespaces);
    validator.finish()
}

fn walk(
    validator: &mut Validator<'_>,
    document: &Document,
    node: NodeId,
    namespaces: &Rc<NamespaceMap>,
) {
    let Some(data) = document.element(node) else {
        return;
    };
    let declarations = document.namespace_declarations(node);
    let namespaces = if declarations.is_empty() {
        Rc::clone(namespaces)
    } else {
        let mut map = (**namespaces).clone();
        for (prefix, namespace) in declarations {
            match prefix {
                Some(prefix) => map.bind(prefix, namespace),
                None => map.set_default(namespace),
            }
        }
        Rc::new(map)
    };

    let mut attributes = Vec::new();
    for attribute in document.attributes(node) {
        let Some(attribute) = document.attr(attribute) else {
            continue;
        };
        if attribute.namespace_uri() == Some(XMLNS_NAMESPACE)
            || attribute.qualified_name() == "xmlns"
        {
            continue;
        }
        attributes.push(Attribute {
            namespace: attribute.namespace_uri().unwrap_or_default(),
            local: attribute.local_name(),
            value: attribute.value(),
        });
    }

    validator.start_element(
        data.namespace_uri().unwrap_or_default(),
        data.local_name(),
        &attributes,
        Rc::clone(&namespaces),
    );

    for child in document.children(node) {
        match document.kind(child) {
            Some(NodeKind::Text | NodeKind::CdataSection) => {
                if let Some(text) = document.character_data(child) {
                    validator.characters(text);
                }
            }
            Some(NodeKind::Element) => walk(validator, document, child, &namespaces),
            _ => {}
        }
    }

    validator.end_element();
}
