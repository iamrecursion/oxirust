//! James Clark's [derivative algorithm for RELAX NG validation][paper].
//!
//! Validation is a fold over the instance: each event turns the current pattern
//! into its *derivative*, the pattern that the rest of the document must match.
//! The document is valid exactly when the final pattern is nullable. The
//! functions here are a direct transcription of the paper — `nullable`,
//! `textDeriv`, `startTagOpenDeriv`, `attDeriv`, `startTagCloseDeriv`,
//! `endTagDeriv`, `applyAfter`, `listDeriv` and `valueMatch` — over the interned
//! [`Arena`](crate::pattern::Arena), which gives the memoization the paper
//! recommends for free.
//!
//! [paper]: https://relaxng.org/jclark/derivative.html

use crate::compile::Schema;
use crate::datatype::DatatypeContext;
use crate::pattern::{Arena, EMPTY, NOT_ALLOWED, PatId, Pattern, TEXT};
use crate::tree::{is_whitespace, split_tokens};

/// The function that [`Engine::apply_after`] pushes through an `after` pattern.
///
/// The paper writes these as partially applied constructors; naming them keeps
/// the derivative allocation-free and lets the arena intern the result.
#[derive(Debug, Clone, Copy)]
enum AfterOp {
    /// `\p -> group p right`
    GroupRight(PatId),
    /// `\p -> interleave p right`
    InterleaveRight(PatId),
    /// `\p -> interleave left p`
    InterleaveLeft(PatId),
    /// `\p -> after p right`
    AfterRight(PatId),
}

/// A derivative engine: the compiled schema plus a working arena that the
/// derivatives extend.
#[derive(Debug)]
pub(crate) struct Engine<'a> {
    schema: &'a Schema,
    arena: Arena,
}

impl<'a> Engine<'a> {
    /// Creates an engine over a compiled schema.
    pub(crate) fn new(schema: &'a Schema) -> Self {
        Self {
            schema,
            arena: schema.arena.clone(),
        }
    }

    /// The `start` pattern of the schema.
    pub(crate) fn start(&self) -> PatId {
        self.schema.start
    }

    /// Whether a pattern matches the empty sequence.
    pub(crate) fn nullable(&self, pattern: PatId) -> bool {
        self.arena.nullable(pattern)
    }

    /// The `choice` constructor, needed by `childrenDeriv`'s whitespace case.
    pub(crate) fn choice(&mut self, left: PatId, right: PatId) -> PatId {
        self.arena.choice(left, right)
    }

    /// Renders the names a pattern would accept next, for diagnostics.
    pub(crate) fn expected(&self, pattern: PatId) -> Vec<String> {
        let mut names = Vec::new();
        self.collect_expected(pattern, &mut names);
        names.sort_unstable();
        names.dedup();
        names
    }

    fn collect_expected(&self, pattern: PatId, names: &mut Vec<String>) {
        match self.arena.get(pattern) {
            Pattern::Element(class, _) | Pattern::Attribute(class, _) => {
                names.push(self.schema.name_classes.describe(*class));
            }
            Pattern::Ref(define) => {
                if let Pattern::Element(class, _) = self.arena.get(self.schema.define(*define)) {
                    names.push(self.schema.name_classes.describe(*class));
                }
            }
            Pattern::Choice(left, right)
            | Pattern::Group(left, right)
            | Pattern::Interleave(left, right)
            | Pattern::After(left, right) => {
                let (left, right) = (*left, *right);
                self.collect_expected(left, names);
                if !matches!(self.arena.get(pattern), Pattern::Group(_, _))
                    || self.arena.nullable(left)
                {
                    self.collect_expected(right, names);
                }
            }
            Pattern::OneOrMore(inner) => {
                let inner = *inner;
                self.collect_expected(inner, names);
            }
            Pattern::Text => names.push("text".to_owned()),
            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // textDeriv
    // -----------------------------------------------------------------------

    /// The derivative of `pattern` with respect to the string `text`.
    pub(crate) fn text_deriv(
        &mut self,
        pattern: PatId,
        text: &str,
        context: &DatatypeContext,
    ) -> PatId {
        match self.arena.get(pattern).clone() {
            Pattern::Choice(left, right) => {
                let left = self.text_deriv(left, text, context);
                let right = self.text_deriv(right, text, context);
                self.arena.choice(left, right)
            }
            Pattern::Interleave(left, right) => {
                let derived_left = self.text_deriv(left, text, context);
                let first = self.arena.interleave(derived_left, right);
                let derived_right = self.text_deriv(right, text, context);
                let second = self.arena.interleave(left, derived_right);
                self.arena.choice(first, second)
            }
            Pattern::Group(left, right) => {
                let derived_left = self.text_deriv(left, text, context);
                let grouped = self.arena.group(derived_left, right);
                if self.arena.nullable(left) {
                    let derived_right = self.text_deriv(right, text, context);
                    self.arena.choice(grouped, derived_right)
                } else {
                    grouped
                }
            }
            Pattern::After(left, right) => {
                let derived = self.text_deriv(left, text, context);
                self.arena.after(derived, right)
            }
            Pattern::OneOrMore(inner) => {
                let derived = self.text_deriv(inner, text, context);
                let optional = self.arena.choice(pattern, EMPTY);
                self.arena.group(derived, optional)
            }
            Pattern::Text => TEXT,
            Pattern::Value {
                datatype,
                value,
                context: value_context,
            } => {
                let Some(datatype) = self.schema.datatypes.get(datatype as usize) else {
                    return NOT_ALLOWED;
                };
                let Some(value_context) = self.schema.contexts.get(value_context as usize) else {
                    return NOT_ALLOWED;
                };
                if datatype.equal(text, context, &value, value_context) {
                    EMPTY
                } else {
                    NOT_ALLOWED
                }
            }
            Pattern::Data { datatype, except } => {
                let Some(datatype) = self.schema.datatypes.get(datatype as usize) else {
                    return NOT_ALLOWED;
                };
                if !datatype.allows(text, context) {
                    return NOT_ALLOWED;
                }
                match except {
                    None => EMPTY,
                    Some(except) => {
                        let derived = self.text_deriv(except, text, context);
                        if self.arena.nullable(derived) {
                            NOT_ALLOWED
                        } else {
                            EMPTY
                        }
                    }
                }
            }
            Pattern::List(inner) => {
                let mut current = inner;
                for token in split_tokens(text) {
                    current = self.text_deriv(current, token, context);
                }
                if self.arena.nullable(current) {
                    EMPTY
                } else {
                    NOT_ALLOWED
                }
            }
            _ => NOT_ALLOWED,
        }
    }

    // -----------------------------------------------------------------------
    // startTagOpenDeriv
    // -----------------------------------------------------------------------

    /// The derivative of `pattern` with respect to an element start tag.
    pub(crate) fn start_tag_open_deriv(
        &mut self,
        pattern: PatId,
        namespace: &str,
        local: &str,
    ) -> PatId {
        match self.arena.get(pattern).clone() {
            Pattern::Choice(left, right) => {
                let left = self.start_tag_open_deriv(left, namespace, local);
                let right = self.start_tag_open_deriv(right, namespace, local);
                self.arena.choice(left, right)
            }
            Pattern::Element(class, content) => {
                if self.schema.name_classes.contains(class, namespace, local) {
                    self.arena.after(content, EMPTY)
                } else {
                    NOT_ALLOWED
                }
            }
            Pattern::Ref(define) => {
                let body = self.schema.define(define);
                self.start_tag_open_deriv(body, namespace, local)
            }
            Pattern::Interleave(left, right) => {
                let derived_left = self.start_tag_open_deriv(left, namespace, local);
                let first = self.apply_after(AfterOp::InterleaveRight(right), derived_left);
                let derived_right = self.start_tag_open_deriv(right, namespace, local);
                let second = self.apply_after(AfterOp::InterleaveLeft(left), derived_right);
                self.arena.choice(first, second)
            }
            Pattern::OneOrMore(inner) => {
                let optional = self.arena.choice(pattern, EMPTY);
                let derived = self.start_tag_open_deriv(inner, namespace, local);
                self.apply_after(AfterOp::GroupRight(optional), derived)
            }
            Pattern::Group(left, right) => {
                let derived_left = self.start_tag_open_deriv(left, namespace, local);
                let first = self.apply_after(AfterOp::GroupRight(right), derived_left);
                if self.arena.nullable(left) {
                    let second = self.start_tag_open_deriv(right, namespace, local);
                    self.arena.choice(first, second)
                } else {
                    first
                }
            }
            Pattern::After(left, right) => {
                let derived = self.start_tag_open_deriv(left, namespace, local);
                self.apply_after(AfterOp::AfterRight(right), derived)
            }
            _ => NOT_ALLOWED,
        }
    }

    fn apply_after(&mut self, op: AfterOp, pattern: PatId) -> PatId {
        match self.arena.get(pattern).clone() {
            Pattern::After(left, right) => {
                let right = self.apply_op(op, right);
                self.arena.after(left, right)
            }
            Pattern::Choice(left, right) => {
                let left = self.apply_after(op, left);
                let right = self.apply_after(op, right);
                self.arena.choice(left, right)
            }
            _ => NOT_ALLOWED,
        }
    }

    fn apply_op(&mut self, op: AfterOp, pattern: PatId) -> PatId {
        match op {
            AfterOp::GroupRight(right) => self.arena.group(pattern, right),
            AfterOp::InterleaveRight(right) => self.arena.interleave(pattern, right),
            AfterOp::InterleaveLeft(left) => self.arena.interleave(left, pattern),
            AfterOp::AfterRight(right) => self.arena.after(pattern, right),
        }
    }

    // -----------------------------------------------------------------------
    // attDeriv
    // -----------------------------------------------------------------------

    /// The derivative of `pattern` with respect to one attribute.
    pub(crate) fn att_deriv(
        &mut self,
        pattern: PatId,
        namespace: &str,
        local: &str,
        value: &str,
        context: &DatatypeContext,
    ) -> PatId {
        match self.arena.get(pattern).clone() {
            Pattern::After(left, right) => {
                let derived = self.att_deriv(left, namespace, local, value, context);
                self.arena.after(derived, right)
            }
            Pattern::Choice(left, right) => {
                let left = self.att_deriv(left, namespace, local, value, context);
                let right = self.att_deriv(right, namespace, local, value, context);
                self.arena.choice(left, right)
            }
            Pattern::Group(left, right) => {
                let derived_left = self.att_deriv(left, namespace, local, value, context);
                let first = self.arena.group(derived_left, right);
                let derived_right = self.att_deriv(right, namespace, local, value, context);
                let second = self.arena.group(left, derived_right);
                self.arena.choice(first, second)
            }
            Pattern::Interleave(left, right) => {
                let derived_left = self.att_deriv(left, namespace, local, value, context);
                let first = self.arena.interleave(derived_left, right);
                let derived_right = self.att_deriv(right, namespace, local, value, context);
                let second = self.arena.interleave(left, derived_right);
                self.arena.choice(first, second)
            }
            Pattern::OneOrMore(inner) => {
                let derived = self.att_deriv(inner, namespace, local, value, context);
                let optional = self.arena.choice(pattern, EMPTY);
                self.arena.group(derived, optional)
            }
            Pattern::Attribute(class, content) => {
                if self.schema.name_classes.contains(class, namespace, local)
                    && self.value_match(content, value, context)
                {
                    EMPTY
                } else {
                    NOT_ALLOWED
                }
            }
            _ => NOT_ALLOWED,
        }
    }

    fn value_match(&mut self, pattern: PatId, value: &str, context: &DatatypeContext) -> bool {
        if self.arena.nullable(pattern) && is_whitespace(value) {
            return true;
        }
        let derived = self.text_deriv(pattern, value, context);
        self.arena.nullable(derived)
    }

    // -----------------------------------------------------------------------
    // startTagCloseDeriv and endTagDeriv
    // -----------------------------------------------------------------------

    /// The derivative taken when the start tag ends: no more attributes may come.
    pub(crate) fn start_tag_close_deriv(&mut self, pattern: PatId) -> PatId {
        match self.arena.get(pattern).clone() {
            Pattern::After(left, right) => {
                let left = self.start_tag_close_deriv(left);
                self.arena.after(left, right)
            }
            Pattern::Choice(left, right) => {
                let left = self.start_tag_close_deriv(left);
                let right = self.start_tag_close_deriv(right);
                self.arena.choice(left, right)
            }
            Pattern::Group(left, right) => {
                let left = self.start_tag_close_deriv(left);
                let right = self.start_tag_close_deriv(right);
                self.arena.group(left, right)
            }
            Pattern::Interleave(left, right) => {
                let left = self.start_tag_close_deriv(left);
                let right = self.start_tag_close_deriv(right);
                self.arena.interleave(left, right)
            }
            Pattern::OneOrMore(inner) => {
                let inner = self.start_tag_close_deriv(inner);
                self.arena.one_or_more(inner)
            }
            Pattern::Attribute(_, _) => NOT_ALLOWED,
            _ => pattern,
        }
    }

    /// The derivative taken when the element ends: resume the `after`
    /// continuation.
    pub(crate) fn end_tag_deriv(&mut self, pattern: PatId) -> PatId {
        match self.arena.get(pattern).clone() {
            Pattern::Choice(left, right) => {
                let left = self.end_tag_deriv(left);
                let right = self.end_tag_deriv(right);
                self.arena.choice(left, right)
            }
            Pattern::After(left, right) => {
                if self.arena.nullable(left) {
                    right
                } else {
                    NOT_ALLOWED
                }
            }
            _ => NOT_ALLOWED,
        }
    }

    /// The continuation an `after` pattern would resume with, ignoring whether
    /// its content matched. Used only to recover after reporting an error, so
    /// that the rest of the document still gets validated.
    pub(crate) fn recovery_continuation(&mut self, pattern: PatId) -> PatId {
        match self.arena.get(pattern).clone() {
            Pattern::Choice(left, right) => {
                let left = self.recovery_continuation(left);
                let right = self.recovery_continuation(right);
                self.arena.choice(left, right)
            }
            Pattern::After(_, right) => right,
            _ => NOT_ALLOWED,
        }
    }
}
