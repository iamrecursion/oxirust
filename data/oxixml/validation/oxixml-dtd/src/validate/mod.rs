//! Checking a document against a [`Dtd`].
//!
//! [`Validator`] is a push interface: drive it with the element and character
//! events of a document — from [`oxixml_xml`], from a DOM walk, or from
//! anything else — and it reports every violated validity constraint of
//! XML 1.0 §2.8, §3 and §4.
//!
//! What it checks:
//!
//! * the document element matches the name in the document type declaration
//!   (VC: Root Element Type);
//! * every element type is declared, and its children match its content model,
//!   using the compiled automaton of [`crate::automaton`] (VC: Element Valid);
//! * every attribute is declared, its value is lexically valid for its declared
//!   type, and `#FIXED` values match (VC: Attribute Value Type, Fixed Attribute
//!   Default);
//! * `#REQUIRED` attributes are present and defaults are supplied for absent
//!   ones (VC: Required Attribute);
//! * `ID` values are unique and every `IDREF`/`IDREFS` token matches one
//!   (VC: ID, IDREF);
//! * `ENTITY`/`ENTITIES` values name declared unparsed entities (VC: Entity
//!   Name) and `NOTATION`/enumeration values are listed (VC: Notation
//!   Attributes, Enumeration);
//! * a `standalone="yes"` declaration is honoured (VC: Standalone Document
//!   Declaration).
//!
//! # Examples
//!
//! ```
//! use oxixml_dtd::{DtdParser, ValidationOptions, Validator};
//!
//! let dtd = DtdParser::new().parse_internal_subset(
//!     r#"<!ELEMENT list (item+)>
//!        <!ELEMENT item (#PCDATA)>
//!        <!ATTLIST item n CDATA #REQUIRED>"#,
//! )?;
//!
//! let mut validator = Validator::new(&dtd, ValidationOptions::default());
//! validator.start_element("list", &[]);
//! validator.start_element("item", &[("n", "1")]);
//! validator.characters("hello");
//! validator.end_element("item");
//! validator.end_element("list");
//! assert!(validator.finish().is_empty());
//! # Ok::<(), oxixml_dtd::DtdError>(())
//! ```

pub(crate) mod attributes;

use std::collections::BTreeSet;

use oxixml_support::pos::TextPosition;

use crate::automaton::{CompiledContent, ContentMatcher};
use crate::error::{ValidityError, ValidityErrorKind};
use crate::model::{AttributeType, ContentSpec, DefaultDecl, Dtd};

pub use attributes::{lexical_error, syntactic_error};

/// How strictly to validate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ValidationOptions {
    /// The value of the document's `standalone` pseudo-attribute, when it had
    /// one. `Some(true)` turns on the standalone-document validity constraint
    /// of XML 1.0 §2.9.
    pub standalone: Option<bool>,
    /// Stop after this many validity errors. `0` means "no limit".
    pub max_errors: usize,
    /// Whether to include the DTD's own
    /// [declaration errors](Dtd::declaration_errors) in the report.
    pub report_declaration_errors: bool,
    /// Whether a non-deterministic content model (XML 1.0 Appendix E) is worth
    /// reporting. Off by default: it is a compatibility requirement rather than
    /// a validity constraint, and many widely deployed DTDs violate it.
    pub report_non_deterministic_models: bool,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            standalone: None,
            max_errors: 0,
            report_declaration_errors: true,
            report_non_deterministic_models: false,
        }
    }
}

impl ValidationOptions {
    /// Options that report everything, including non-deterministic content
    /// models.
    #[must_use]
    pub fn pedantic() -> Self {
        Self {
            standalone: None,
            max_errors: 0,
            report_declaration_errors: true,
            report_non_deterministic_models: true,
        }
    }

    /// Sets the `standalone` pseudo-attribute.
    #[must_use]
    pub fn with_standalone(mut self, standalone: Option<bool>) -> Self {
        self.standalone = standalone;
        self
    }
}

/// One open element.
struct Frame<'a> {
    name: String,
    matcher: Option<ContentMatcher<'a>>,
    /// Whether the element type is declared with element content, where white
    /// space is not significant.
    element_content: bool,
    /// Whether the element type declaration came from the external subset.
    declared_externally: bool,
}

/// Checks a stream of document events against a [`Dtd`].
pub struct Validator<'a> {
    dtd: &'a Dtd,
    options: ValidationOptions,
    stack: Vec<Frame<'a>>,
    ids: BTreeSet<String>,
    idrefs: Vec<(String, TextPosition)>,
    errors: Vec<ValidityError>,
    position: TextPosition,
    root_seen: bool,
}

impl<'a> Validator<'a> {
    /// Starts validating against `dtd`.
    #[must_use]
    pub fn new(dtd: &'a Dtd, options: ValidationOptions) -> Self {
        let mut validator = Self {
            dtd,
            options,
            stack: Vec::new(),
            ids: BTreeSet::new(),
            idrefs: Vec::new(),
            errors: Vec::new(),
            position: TextPosition::start(),
            root_seen: false,
        };
        if options.report_declaration_errors {
            for error in dtd.declaration_errors() {
                if matches!(
                    error.kind(),
                    ValidityErrorKind::NonDeterministicContentModel { .. }
                ) && !options.report_non_deterministic_models
                {
                    continue;
                }
                validator.errors.push(error.clone());
            }
        }
        validator
    }

    /// Sets the position reported for subsequent errors.
    ///
    /// `position` is 0-based, as stored by [`oxixml_support::pos::TextPosition`].
    pub fn set_position(&mut self, position: TextPosition) {
        self.position = position;
    }

    /// The errors found so far.
    #[must_use]
    pub fn errors(&self) -> &[ValidityError] {
        &self.errors
    }

    /// Whether any error has been found.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Records a violated constraint.
    fn report(&mut self, kind: ValidityErrorKind) {
        if self.options.max_errors > 0 && self.errors.len() >= self.options.max_errors {
            return;
        }
        self.errors.push(ValidityError::new(kind, self.position));
    }

    /// Opens an element.
    ///
    /// `attributes` are the attribute names paired with their values *after*
    /// the `CDATA` half of attribute-value normalization (XML 1.0 §3.3.3) —
    /// that is, with references expanded and literal white space turned into
    /// spaces, but with no collapsing yet. The extra collapsing that tokenized
    /// types call for is applied here, where the declared type is known.
    ///
    /// Returns the attributes that must be *added* to the element because the
    /// DTD declares a default for them.
    pub fn start_element(
        &mut self,
        name: &str,
        attributes: &[(&str, &str)],
    ) -> Vec<(String, String)> {
        self.check_parent_accepts(name);
        self.check_root(name);

        let declaration = self.dtd.element(name);
        if declaration.is_none() {
            self.report(ValidityErrorKind::ElementNotDeclared(name.to_owned()));
        }
        let element_content =
            declaration.is_some_and(|d| matches!(d.content(), ContentSpec::Children(_)));
        let declared_externally = declaration.is_some_and(crate::model::ElementDecl::is_external);
        let matcher = self
            .dtd
            .compiled_content(name)
            .map(CompiledContent::matcher);

        let defaults = self.check_attributes(name, attributes);

        self.stack.push(Frame {
            name: name.to_owned(),
            matcher,
            element_content,
            declared_externally,
        });
        defaults
    }

    /// VC: Element Valid — offers `name` to the parent's content model.
    fn check_parent_accepts(&mut self, name: &str) {
        let Some(parent) = self.stack.last_mut() else {
            return;
        };
        let parent_name = parent.name.clone();
        let Some(matcher) = parent.matcher.as_mut() else {
            return;
        };
        if matcher.has_failed() {
            // The content model already went wrong; reporting every subsequent
            // child would bury the one error that matters.
            return;
        }
        if matcher.accept(name) {
            return;
        }
        let expected = matcher.expected();
        let empty = matches!(
            self.dtd.compiled_content(&parent_name),
            Some(CompiledContent::Empty)
        );
        if empty {
            self.report(ValidityErrorKind::EmptyElementNotEmpty {
                element: parent_name,
            });
        } else {
            self.report(ValidityErrorKind::ContentModelMismatch {
                element: parent_name,
                found: Some(name.to_owned()),
                expected,
            });
        }
    }

    /// VC: Root Element Type.
    fn check_root(&mut self, name: &str) {
        if self.root_seen || !self.stack.is_empty() {
            return;
        }
        self.root_seen = true;
        if let Some(declared) = self.dtd.name()
            && declared != name
        {
            self.report(ValidityErrorKind::RootElementType {
                declared: declared.to_owned(),
                found: name.to_owned(),
            });
        }
    }

    /// Checks supplied attributes and computes the defaulted ones.
    fn check_attributes(
        &mut self,
        element: &str,
        attributes: &[(&str, &str)],
    ) -> Vec<(String, String)> {
        let list = self.dtd.attributes(element);
        for (name, value) in attributes {
            let Some(declaration) = list.and_then(|list| list.get(name)) else {
                self.report(ValidityErrorKind::AttributeNotDeclared {
                    element: element.to_owned(),
                    attribute: (*name).to_owned(),
                });
                continue;
            };
            let att_type = declaration.att_type().clone();
            let default = declaration.default().clone();
            let external = declaration.is_external();
            let normalized = if att_type.is_tokenized() {
                collapse(value)
            } else {
                (*value).to_owned()
            };

            if external
                && self.options.standalone == Some(true)
                && att_type.is_tokenized()
                && normalized != *value
            {
                self.report(ValidityErrorKind::StandaloneViolation {
                    detail: format!(
                        "attribute `{name}` of `{element}` is declared in the external subset \
                         and its value changes under normalization"
                    ),
                });
            }

            if let Some(reason) = lexical_error(self.dtd, &att_type, &normalized) {
                self.report(ValidityErrorKind::AttributeValueInvalid {
                    element: element.to_owned(),
                    attribute: (*name).to_owned(),
                    value: normalized.clone(),
                    reason,
                });
            }
            if let DefaultDecl::Fixed(expected) = &default
                && *expected != normalized
            {
                self.report(ValidityErrorKind::FixedAttributeMismatch {
                    element: element.to_owned(),
                    attribute: (*name).to_owned(),
                    expected: expected.clone(),
                    found: normalized.clone(),
                });
            }
            self.record_identity(element, &att_type, &normalized);
        }

        self.apply_defaults(element, attributes)
    }

    /// Tracks `ID` uniqueness and collects `IDREF`s for the final check.
    fn record_identity(&mut self, element: &str, att_type: &AttributeType, value: &str) {
        match att_type {
            AttributeType::Id if !self.ids.insert(value.to_owned()) => {
                self.report(ValidityErrorKind::DuplicateId {
                    value: value.to_owned(),
                    element: element.to_owned(),
                });
            }
            AttributeType::IdRef => self.idrefs.push((value.to_owned(), self.position)),
            AttributeType::IdRefs => {
                for token in value.split(' ').filter(|token| !token.is_empty()) {
                    self.idrefs.push((token.to_owned(), self.position));
                }
            }
            _ => {}
        }
    }

    /// VC: Required Attribute, plus attribute defaulting and the standalone
    /// constraint on externally declared defaults.
    fn apply_defaults(
        &mut self,
        element: &str,
        attributes: &[(&str, &str)],
    ) -> Vec<(String, String)> {
        let Some(list) = self.dtd.attributes(element) else {
            return Vec::new();
        };
        let mut added = Vec::new();
        let mut violations = Vec::new();
        for declaration in list.iter() {
            if attributes
                .iter()
                .any(|(name, _)| *name == declaration.name())
            {
                continue;
            }
            match declaration.default() {
                DefaultDecl::Required => {
                    violations.push(ValidityErrorKind::RequiredAttributeMissing {
                        element: element.to_owned(),
                        attribute: declaration.name().to_owned(),
                    });
                }
                DefaultDecl::Implied => {}
                DefaultDecl::Fixed(value) | DefaultDecl::Default(value) => {
                    if declaration.is_external() && self.options.standalone == Some(true) {
                        violations.push(ValidityErrorKind::StandaloneViolation {
                            detail: format!(
                                "attribute `{}` of `{element}` takes a default value from the \
                                 external subset",
                                declaration.name()
                            ),
                        });
                    }
                    added.push((declaration.name().to_owned(), value.clone()));
                }
            }
        }
        let identities: Vec<(AttributeType, String)> = added
            .iter()
            .filter_map(|(name, value)| {
                list.get(name)
                    .map(|declaration| (declaration.att_type().clone(), value.clone()))
            })
            .collect();
        for violation in violations {
            self.report(violation);
        }
        for (att_type, value) in identities {
            self.record_identity(element, &att_type, &value);
        }
        added
    }

    /// Reports literal character data appearing in the current element.
    ///
    /// White space written literally in element content is ignorable and so is
    /// permitted; anything else in element content, and *any* content at all in
    /// an `EMPTY` element, is a violation.
    pub fn characters(&mut self, text: &str) {
        self.content(text, true);
    }

    /// Reports character data that came from a character reference, a CDATA
    /// section, or anywhere else that is never *ignorable* white space.
    ///
    /// XML 1.0 erratum E15 is explicit about this: `<foo><bar/>&#32;<bar/></foo>`
    /// is invalid when `foo` has element content, even though the very same
    /// space written literally would have been ignorable.
    pub fn reference_data(&mut self, text: &str) {
        self.content(text, false);
    }

    /// Reports a construct that makes an element non-empty without necessarily
    /// producing character data: a comment, a processing instruction, a CDATA
    /// section, or a general-entity reference.
    ///
    /// Only an `EMPTY` element cares — erratum E15 makes
    /// `<foo><!-- c --></foo>` invalid when `foo` is declared `EMPTY`.
    pub fn markup(&mut self) {
        let Some(frame) = self.stack.last() else {
            return;
        };
        let element = frame.name.clone();
        if matches!(
            self.dtd.compiled_content(&element),
            Some(CompiledContent::Empty)
        ) {
            self.report(ValidityErrorKind::EmptyElementNotEmpty { element });
        }
    }

    /// Shared implementation of [`characters`](Self::characters) and
    /// [`reference_data`](Self::reference_data).
    fn content(&mut self, text: &str, white_space_is_ignorable: bool) {
        let Some(frame) = self.stack.last() else {
            return;
        };
        let element = frame.name.clone();
        let element_content = frame.element_content;
        let declared_externally = frame.declared_externally;
        let is_empty = matches!(
            self.dtd.compiled_content(&element),
            Some(CompiledContent::Empty)
        );
        let blank =
            white_space_is_ignorable && text.chars().all(|c| matches!(c, ' ' | '\t' | '\r' | '\n'));

        if is_empty {
            if !text.is_empty() {
                self.report(ValidityErrorKind::EmptyElementNotEmpty { element });
            }
            return;
        }
        if !element_content {
            return;
        }
        if blank {
            if declared_externally && self.options.standalone == Some(true) && !text.is_empty() {
                self.report(ValidityErrorKind::StandaloneViolation {
                    detail: format!(
                        "white space appears directly in `{element}`, whose element content is \
                         declared in the external subset"
                    ),
                });
            }
            return;
        }
        if text.is_empty() {
            return;
        }
        self.report(ValidityErrorKind::CharacterDataNotAllowed { element });
    }

    /// Closes the current element, checking that its content model is satisfied.
    pub fn end_element(&mut self, _name: &str) {
        let Some(frame) = self.stack.pop() else {
            return;
        };
        let Some(matcher) = frame.matcher else {
            return;
        };
        if matcher.is_final() || matcher.has_failed() {
            return;
        }
        self.report(ValidityErrorKind::ContentModelMismatch {
            element: frame.name,
            found: None,
            expected: matcher.expected(),
        });
    }

    /// Records a violation found by the caller — for instance a reference to an
    /// entity declared in the external subset of a `standalone="yes"` document.
    pub fn report_error(&mut self, kind: ValidityErrorKind) {
        self.report(kind);
    }

    /// Finishes validation, resolving `IDREF`s, and returns every violation
    /// found.
    #[must_use]
    pub fn finish(mut self) -> Vec<ValidityError> {
        let dangling: Vec<(String, TextPosition)> = self
            .idrefs
            .iter()
            .filter(|(value, _)| !self.ids.contains(value))
            .cloned()
            .collect();
        for (value, position) in dangling {
            self.position = position;
            self.report(ValidityErrorKind::UnresolvedIdref { value });
        }
        self.errors
    }
}

/// The tokenized half of attribute-value normalization (XML 1.0 §3.3.3):
/// discard leading and trailing spaces, and collapse internal runs.
#[must_use]
pub fn collapse(value: &str) -> String {
    let parts: Vec<&str> = value.split(' ').filter(|part| !part.is_empty()).collect();
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DtdParser;

    fn dtd(text: &str) -> Dtd {
        DtdParser::new()
            .parse_internal_subset(text)
            .expect("valid DTD")
    }

    #[test]
    fn a_conforming_document_produces_no_errors() {
        let dtd = dtd("<!ELEMENT a (b, c)><!ELEMENT b EMPTY><!ELEMENT c (#PCDATA)>");
        let mut validator = Validator::new(&dtd, ValidationOptions::default());
        validator.start_element("a", &[]);
        validator.start_element("b", &[]);
        validator.end_element("b");
        validator.start_element("c", &[]);
        validator.characters("text");
        validator.end_element("c");
        validator.end_element("a");
        assert_eq!(validator.finish(), Vec::new());
    }

    #[test]
    fn out_of_order_children_are_rejected() {
        let dtd = dtd("<!ELEMENT a (b, c)><!ELEMENT b EMPTY><!ELEMENT c EMPTY>");
        let mut validator = Validator::new(&dtd, ValidationOptions::default());
        validator.start_element("a", &[]);
        validator.start_element("c", &[]);
        validator.end_element("c");
        validator.end_element("a");
        let errors = validator.finish();
        assert!(matches!(
            errors.first().map(ValidityError::kind),
            Some(ValidityErrorKind::ContentModelMismatch { found: Some(_), .. })
        ));
    }

    #[test]
    fn an_incomplete_content_model_is_reported_at_the_end_tag() {
        let dtd = dtd("<!ELEMENT a (b, c)><!ELEMENT b EMPTY><!ELEMENT c EMPTY>");
        let mut validator = Validator::new(&dtd, ValidationOptions::default());
        validator.start_element("a", &[]);
        validator.start_element("b", &[]);
        validator.end_element("b");
        validator.end_element("a");
        let errors = validator.finish();
        assert!(matches!(
            errors.first().map(ValidityError::kind),
            Some(ValidityErrorKind::ContentModelMismatch { found: None, .. })
        ));
    }

    #[test]
    fn character_data_is_rejected_in_element_content_but_white_space_is_not() {
        let dtd = dtd("<!ELEMENT a (b)><!ELEMENT b EMPTY>");
        let mut validator = Validator::new(&dtd, ValidationOptions::default());
        validator.start_element("a", &[]);
        validator.characters("\n  ");
        validator.start_element("b", &[]);
        validator.end_element("b");
        validator.characters("oops");
        validator.end_element("a");
        let errors = validator.finish();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0].kind(),
            ValidityErrorKind::CharacterDataNotAllowed { .. }
        ));
    }

    #[test]
    fn empty_elements_may_not_have_content() {
        let dtd = dtd("<!ELEMENT a EMPTY>");
        let mut validator = Validator::new(&dtd, ValidationOptions::default());
        validator.start_element("a", &[]);
        validator.characters(" ");
        validator.end_element("a");
        assert!(matches!(
            validator.finish().first().map(ValidityError::kind),
            Some(ValidityErrorKind::EmptyElementNotEmpty { .. })
        ));
    }

    #[test]
    fn required_attributes_are_enforced_and_defaults_supplied() {
        let dtd = dtd(
            r#"<!ELEMENT a EMPTY><!ATTLIST a r CDATA #REQUIRED d CDATA "fallback" f CDATA #FIXED "k">"#,
        );
        let mut validator = Validator::new(&dtd, ValidationOptions::default());
        let defaults = validator.start_element("a", &[]);
        validator.end_element("a");
        assert_eq!(
            defaults,
            vec![
                ("d".to_owned(), "fallback".to_owned()),
                ("f".to_owned(), "k".to_owned())
            ]
        );
        assert!(matches!(
            validator.finish().first().map(ValidityError::kind),
            Some(ValidityErrorKind::RequiredAttributeMissing { .. })
        ));
    }

    #[test]
    fn fixed_attributes_must_match_their_declared_value() {
        let dtd = dtd(r#"<!ELEMENT a EMPTY><!ATTLIST a f CDATA #FIXED "k">"#);
        let mut validator = Validator::new(&dtd, ValidationOptions::default());
        validator.start_element("a", &[("f", "other")]);
        validator.end_element("a");
        assert!(matches!(
            validator.finish().first().map(ValidityError::kind),
            Some(ValidityErrorKind::FixedAttributeMismatch { .. })
        ));
    }

    #[test]
    fn ids_must_be_unique_and_idrefs_must_resolve() {
        let dtd = dtd(
            "<!ELEMENT a (b+)><!ELEMENT b EMPTY><!ATTLIST b id ID #IMPLIED ref IDREF #IMPLIED>",
        );
        let mut validator = Validator::new(&dtd, ValidationOptions::default());
        validator.start_element("a", &[]);
        validator.start_element("b", &[("id", "x")]);
        validator.end_element("b");
        validator.start_element("b", &[("id", "x")]);
        validator.end_element("b");
        validator.start_element("b", &[("ref", "missing")]);
        validator.end_element("b");
        validator.end_element("a");
        let kinds: Vec<_> = validator
            .finish()
            .into_iter()
            .map(ValidityError::into_kind)
            .collect();
        assert!(
            kinds
                .iter()
                .any(|kind| matches!(kind, ValidityErrorKind::DuplicateId { .. }))
        );
        assert!(
            kinds
                .iter()
                .any(|kind| matches!(kind, ValidityErrorKind::UnresolvedIdref { .. }))
        );
    }

    #[test]
    fn tokenized_attribute_values_are_collapsed_before_checking() {
        let dtd = dtd("<!ELEMENT a EMPTY><!ATTLIST a t NMTOKENS #IMPLIED>");
        let mut validator = Validator::new(&dtd, ValidationOptions::default());
        validator.start_element("a", &[("t", "  x   y  ")]);
        validator.end_element("a");
        assert!(validator.finish().is_empty());
        assert_eq!(collapse("  x   y  "), "x y");
    }

    #[test]
    fn undeclared_elements_and_attributes_are_reported() {
        let dtd = dtd("<!ELEMENT a EMPTY>");
        let mut validator = Validator::new(&dtd, ValidationOptions::default());
        validator.start_element("a", &[("nope", "1")]);
        validator.end_element("a");
        validator.start_element("ghost", &[]);
        validator.end_element("ghost");
        let kinds: Vec<_> = validator
            .finish()
            .into_iter()
            .map(ValidityError::into_kind)
            .collect();
        assert!(
            kinds
                .iter()
                .any(|kind| matches!(kind, ValidityErrorKind::AttributeNotDeclared { .. }))
        );
        assert!(
            kinds
                .iter()
                .any(|kind| matches!(kind, ValidityErrorKind::ElementNotDeclared(_)))
        );
    }

    #[test]
    fn standalone_documents_may_not_take_defaults_from_the_external_subset() {
        let dtd = DtdParser::new()
            .parse_external_subset(r#"<!ELEMENT a EMPTY><!ATTLIST a d CDATA "x">"#)
            .expect("valid DTD");
        let options = ValidationOptions::default().with_standalone(Some(true));
        let mut validator = Validator::new(&dtd, options);
        validator.start_element("a", &[]);
        validator.end_element("a");
        assert!(matches!(
            validator.finish().first().map(ValidityError::kind),
            Some(ValidityErrorKind::StandaloneViolation { .. })
        ));
    }
}
