//! Validating a whole XML document, end to end.
//!
//! [`DocumentValidator`] wires the three pieces of this crate together with the
//! pull parser of [`oxixml_xml`]:
//!
//! 1. read the document type declaration and build a [`Dtd`] from it (internal
//!    subset first, then — only if a resolver allows it — the external subset);
//! 2. replay the document through a [`Validator`];
//! 3. expand general-entity references *as content*, recursively, so that
//!    markup inside an entity is validated exactly as if it had been written
//!    out. Every expansion is charged against the same
//!    [`Budget`], which is what stops an expansion bomb here as
//!    well as in the DTD.
//!
//! # Examples
//!
//! ```
//! use oxixml_dtd::validate_str;
//!
//! let report = validate_str(
//!     r#"<?xml version="1.0"?>
//!        <!DOCTYPE list [
//!          <!ELEMENT list (item+)>
//!          <!ELEMENT item (#PCDATA)>
//!          <!ATTLIST item id ID #REQUIRED>
//!        ]>
//!        <list><item id="a">one</item><item id="b">two</item></list>"#,
//! )?;
//! assert!(report.is_valid());
//!
//! let report = validate_str(
//!     r#"<!DOCTYPE list [<!ELEMENT list (item+)><!ELEMENT item EMPTY>]>
//!        <list><other/></list>"#,
//! )?;
//! assert!(!report.is_valid());
//! # Ok::<(), oxixml_dtd::Error>(())
//! ```

use oxixml_xml::events::{BytesRef, BytesStart, Event};
use oxixml_xml::reader::Reader;

use crate::entity::builtin_entity;
use crate::error::{DtdError, DtdErrorKind, Error, ValidityError, ValidityErrorKind};
use crate::limits::{Budget, ExpansionLimits};
use crate::model::{Dtd, EntityDefinition};
use crate::parser::DtdParser;
use crate::resolver::{EntityResolver, ExternalKind, ResolveRequest, ResolverError};
use crate::validate::{ValidationOptions, Validator};

/// The outcome of validating a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    dtd: Dtd,
    errors: Vec<ValidityError>,
    standalone: Option<bool>,
}

impl ValidationReport {
    /// Whether the document satisfied every validity constraint.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Every violation found, in document order.
    #[must_use]
    pub fn errors(&self) -> &[ValidityError] {
        &self.errors
    }

    /// The DTD the document was checked against.
    #[must_use]
    pub fn dtd(&self) -> &Dtd {
        &self.dtd
    }

    /// The value of the document's `standalone` pseudo-attribute.
    #[must_use]
    pub fn standalone(&self) -> Option<bool> {
        self.standalone
    }

    /// Splits the report into its DTD and its errors.
    #[must_use]
    pub fn into_parts(self) -> (Dtd, Vec<ValidityError>) {
        (self.dtd, self.errors)
    }
}

/// Reads a document, its DTD, and checks one against the other.
///
/// Like [`DtdParser`], it fetches nothing external unless a resolver is
/// installed.
pub struct DocumentValidator {
    parser: DtdParser,
    options: ValidationOptions,
    limits: ExpansionLimits,
}

impl Default for DocumentValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for DocumentValidator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DocumentValidator")
            .field("options", &self.options)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl DocumentValidator {
    /// Creates a validator with the default options and no external access.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parser: DtdParser::new(),
            options: ValidationOptions::default(),
            limits: ExpansionLimits::DEFAULT,
        }
    }

    /// Installs an [`EntityResolver`], enabling the external subset and
    /// external entities.
    #[must_use]
    pub fn with_resolver(mut self, resolver: Box<dyn EntityResolver>) -> Self {
        self.parser = self.parser.with_resolver(resolver);
        self
    }

    /// Sets the entity-expansion budget, for both the DTD and the document.
    #[must_use]
    pub fn with_limits(mut self, limits: ExpansionLimits) -> Self {
        self.limits = limits;
        self.parser = self.parser.with_limits(limits);
        self
    }

    /// Sets the validation options. A `standalone` set here is overridden by
    /// the document's own XML declaration when it has one.
    #[must_use]
    pub fn with_options(mut self, options: ValidationOptions) -> Self {
        self.options = options;
        self
    }

    /// How much of the expansion budget the last call to
    /// [`validate_str`](Self::validate_str) (or a `_with_base` variant) consumed, across *both*
    /// the DTD and the document's content — the same combined total
    /// [`DtdParser::budget`](crate::DtdParser::budget) reports for the parser this validator
    /// drives internally.
    #[must_use]
    pub fn budget(&self) -> Budget {
        self.parser.budget()
    }

    /// Validates `xml`.
    ///
    /// # Errors
    ///
    /// [`Error::Xml`] when the document is not well formed, or [`Error::Dtd`]
    /// when the DTD cannot be read — including
    /// [`DtdErrorKind::EntityExpansionLimit`] when the document is an expansion
    /// bomb. Validity *errors* are returned in the report rather than as an
    /// `Err`.
    pub fn validate_str(&mut self, xml: &str) -> Result<ValidationReport, Error> {
        self.validate_str_with_base(xml, None)
    }

    /// [`validate_str`](Self::validate_str), with the document's base URI so
    /// that relative system identifiers resolve.
    ///
    /// # Errors
    ///
    /// As [`validate_str`](Self::validate_str).
    pub fn validate_str_with_base(
        &mut self,
        xml: &str,
        base: Option<&str>,
    ) -> Result<ValidationReport, Error> {
        let (dtd, standalone) = self.read_prolog(xml, base)?;
        let mut options = self.options;
        if standalone.is_some() {
            options.standalone = standalone;
        }
        // The content phase continues spending the *same* budget the DTD phase already
        // charged against, rather than starting over with a fresh full allowance — otherwise
        // the two phases would each silently get their own complete budget, doubling (or, with
        // more phases, multiplying) the total an attacker could spend before either one ever
        // reports `EntityExpansionLimit`.
        let (errors, spent) = {
            let mut run = Run {
                validator: Validator::new(&dtd, options),
                dtd: &dtd,
                budget: self.parser.budget(),
                entity_stack: Vec::new(),
                standalone: options.standalone,
                resolver: self.parser.resolver_mut(),
            };
            run.feed(xml, true)?;
            (run.validator.finish(), run.budget)
        };
        // Written back so that `self.parser.budget()` — and so
        // [`DtdParser::budget`](crate::DtdParser::budget) for anyone driving this validator and
        // then inspecting the parser directly — reports the combined DTD-and-content total,
        // matching what it already claims to track.
        self.parser.set_budget(spent);
        Ok(ValidationReport {
            dtd,
            errors,
            standalone,
        })
    }

    /// Reads the XML declaration and the document type declaration.
    fn read_prolog(&mut self, xml: &str, base: Option<&str>) -> Result<(Dtd, Option<bool>), Error> {
        let mut reader = Reader::from_str(xml);
        let mut standalone = None;
        loop {
            match reader
                .read_event()
                .map_err(|error| Error::Xml(error.to_string()))?
            {
                Event::Decl(declaration) => {
                    if let Some(value) = declaration.standalone() {
                        let value = value.map_err(|error| Error::Xml(error.to_string()))?;
                        standalone = match value.as_ref() {
                            b"yes" => Some(true),
                            b"no" => Some(false),
                            _ => None,
                        };
                    }
                }
                Event::DocType(text) => {
                    let body = text
                        .decode()
                        .map_err(|error| Error::Xml(error.to_string()))?;
                    let dtd = self.parser.parse_doctype_with_base(&body, base)?;
                    return Ok((dtd, standalone));
                }
                Event::Start(_) | Event::Empty(_) | Event::Eof => {
                    // A document with no document type declaration never calls
                    // `parse_doctype_with_base`, which is otherwise what resets
                    // `self.parser`'s budget for this call. Reset it here instead, so that
                    // a `DocumentValidator` reused across several documents never lets a
                    // doctype-less document silently inherit — and validate its own content
                    // against — whatever budget an *earlier* call happened to leave behind.
                    self.parser.set_budget(self.limits.budget());
                    return Ok((Dtd::new(), standalone));
                }
                _ => {}
            }
        }
    }
}

/// Validates `xml` with the default options: no external access, default
/// expansion limits.
///
/// # Errors
///
/// As [`DocumentValidator::validate_str`].
pub fn validate_str(xml: &str) -> Result<ValidationReport, Error> {
    DocumentValidator::new().validate_str(xml)
}

/// One pass of a document (or of one entity's replacement text) through the
/// validator.
struct Run<'a> {
    validator: Validator<'a>,
    dtd: &'a Dtd,
    budget: Budget,
    entity_stack: Vec<String>,
    standalone: Option<bool>,
    resolver: &'a mut dyn EntityResolver,
}

impl Run<'_> {
    /// Replays `text` as XML content through the validator.
    ///
    /// `document` distinguishes the document itself — which has a prolog — from
    /// the replacement text of an entity, which does not.
    fn feed(&mut self, text: &str, document: bool) -> Result<(), Error> {
        let mut reader = Reader::from_str(text);
        reader.config_mut().expand_empty_elements = true;
        loop {
            let event = reader
                .read_event()
                .map_err(|error| Error::Xml(error.to_string()))?;
            match event {
                Event::Start(start) => self.start_element(&start)?,
                Event::End(end) => {
                    let name = decode(end.name().as_ref())?;
                    self.validator.end_element(&name);
                }
                Event::Text(chunk) => {
                    let text = chunk
                        .decode()
                        .map_err(|error| Error::Xml(error.to_string()))?;
                    self.validator.characters(&text);
                }
                Event::CData(chunk) => {
                    // A CDATA section is character data even when it is empty,
                    // and its white space is never ignorable (erratum E15).
                    let text = chunk
                        .decode()
                        .map_err(|error| Error::Xml(error.to_string()))?;
                    self.validator.markup();
                    self.validator.reference_data(&text);
                }
                Event::Comment(_) | Event::PI(_) => self.validator.markup(),
                Event::GeneralRef(reference) => self.general_reference(&reference)?,
                Event::Eof => return Ok(()),
                Event::Decl(_) if !document => {
                    // A text declaration at the head of an external entity.
                }
                _ => {}
            }
        }
    }

    /// Feeds one start tag, after normalizing its attribute values.
    fn start_element(&mut self, start: &BytesStart<'_>) -> Result<(), Error> {
        let name = decode(start.name().as_ref())?;
        let mut attributes: Vec<(String, String)> = Vec::new();
        for attribute in start.attributes() {
            let attribute = attribute.map_err(|error| Error::Xml(error.to_string()))?;
            let key = decode(attribute.key.as_ref())?;
            let raw = decode(&attribute.value)?;
            let value = match self
                .dtd
                .normalize_attribute_value(&raw, false, &mut self.budget)
            {
                Ok(value) => value,
                Err(error) => match error.kind() {
                    DtdErrorKind::UndeclaredGeneralEntity(entity) => {
                        let entity = entity.clone();
                        self.validator
                            .report_error(ValidityErrorKind::EntityNotDeclared { name: entity });
                        raw
                    }
                    _ => return Err(Error::from(error)),
                },
            };
            attributes.push((key, value));
        }
        let borrowed: Vec<(&str, &str)> = attributes
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        self.validator.start_element(&name, &borrowed);
        Ok(())
    }

    /// Expands a general-entity or character reference found in content.
    fn general_reference(&mut self, reference: &BytesRef<'_>) -> Result<(), Error> {
        if reference.is_char_ref() {
            let resolved = reference
                .resolve_char_ref()
                .map_err(|error| Error::Xml(error.to_string()))?;
            if let Some(c) = resolved {
                // Erratum E15: a character reference is never ignorable white
                // space, even when it denotes one.
                self.validator.reference_data(c.encode_utf8(&mut [0_u8; 4]));
            }
            return Ok(());
        }
        let name = decode(reference.as_ref())?;
        // Erratum E15: the reference itself makes an `EMPTY` element non-empty,
        // whatever it expands to.
        self.validator.markup();
        if let Some(builtin) = builtin_entity(&name) {
            self.validator.reference_data(builtin);
            return Ok(());
        }
        let Some(entity) = self.dtd.general_entity(&name) else {
            self.validator
                .report_error(ValidityErrorKind::EntityNotDeclared { name });
            return Ok(());
        };
        if entity.is_unparsed() {
            return Err(Error::from(DtdError::detached(
                DtdErrorKind::UnparsedEntityReference(name),
            )));
        }
        if self.standalone == Some(true) && entity.is_declared_externally() {
            self.validator
                .report_error(ValidityErrorKind::StandaloneViolation {
                    detail: format!("entity `{name}` is declared in the external subset"),
                });
        }
        if self.entity_stack.contains(&name) {
            return Err(Error::from(DtdError::detached(
                DtdErrorKind::RecursiveEntity {
                    chain: self.entity_stack.clone(),
                    name,
                },
            )));
        }

        let declared_base = entity.base().map(str::to_owned);
        let text = match entity.definition() {
            EntityDefinition::Internal(text) => text.clone(),
            EntityDefinition::External { id, .. } => {
                let system_id = id.system_id().unwrap_or_default().to_owned();
                let public_id = id.public_id().map(str::to_owned);
                let request = ResolveRequest {
                    kind: ExternalKind::GeneralEntity,
                    name: Some(&name),
                    public_id: public_id.as_deref(),
                    system_id: &system_id,
                    base: declared_base.as_deref(),
                };
                match self.resolver.resolve(&request) {
                    Ok(resolved) => resolved.text,
                    Err(ResolverError::Denied) => {
                        return Err(Error::from(DtdError::detached(
                            DtdErrorKind::ExternalEntityDenied { system_id },
                        )));
                    }
                    Err(ResolverError::Unavailable(reason)) => {
                        return Err(Error::from(DtdError::detached(
                            DtdErrorKind::ExternalEntityUnavailable { system_id, reason },
                        )));
                    }
                }
            }
        };

        self.budget
            .check_depth(self.entity_stack.len() + 1)
            .map_err(Error::from)?;
        self.budget
            .charge_expansion(text.chars().count())
            .map_err(Error::from)?;
        self.entity_stack.push(name);
        let result = self.feed(&text, false);
        self.entity_stack.pop();
        result
    }
}

/// Decodes UTF-8 bytes coming out of the reader.
fn decode(bytes: &[u8]) -> Result<String, Error> {
    core::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|error| Error::Xml(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolver::MemoryResolver;

    #[test]
    fn a_valid_document_reports_no_errors() -> Result<(), Error> {
        let report = validate_str(
            r#"<!DOCTYPE a [<!ELEMENT a (b)><!ELEMENT b (#PCDATA)>]><a><b>x</b></a>"#,
        )?;
        assert!(report.is_valid());
        assert_eq!(report.dtd().name(), Some("a"));
        Ok(())
    }

    #[test]
    fn budget_reports_the_dtd_and_content_expansion_combined() -> Result<(), Error> {
        let mut validator = DocumentValidator::new();
        validator.validate_str(
            r#"<!DOCTYPE a [
                 <!ELEMENT a (#PCDATA)>
                 <!ENTITY default_bomb "aaaa">
                 <!ATTLIST a attr CDATA "&default_bomb;">
                 <!ENTITY content_bomb "bbbbbb">
               ]><a>&content_bomb;</a>"#,
        )?;
        let budget = validator.budget();
        // One expansion while normalizing the `<!ATTLIST>` default (4 characters) and one while
        // validating the document's content (6 characters): both phases' cost must show up in
        // the same total.
        assert_eq!(budget.expansions(), 2);
        assert_eq!(budget.expanded_size(), 10);
        Ok(())
    }

    #[test]
    fn a_doctype_less_document_does_not_inherit_a_reused_validators_earlier_budget()
    -> Result<(), Error> {
        // `read_prolog` only resets `self.parser`'s budget by way of
        // `parse_doctype_with_base`, which a document with no `<!DOCTYPE …>` never calls. A
        // `DocumentValidator` is explicitly reusable across several `validate_str` calls
        // (`&mut self`, not `self`), so the second, doctype-less call here must still see a
        // fresh budget rather than silently continuing to spend the first call's.
        let mut validator = DocumentValidator::new();
        validator.validate_str(
            r#"<!DOCTYPE a [
                 <!ELEMENT a (#PCDATA)>
                 <!ENTITY bomb "aaaaaaaaaa">
               ]><a>&bomb;</a>"#,
        )?;
        assert_eq!(validator.budget().expanded_size(), 10);

        // No document type declaration at all, and no entity references, so this call's own
        // expansion cost is genuinely zero — `budget()` must reflect exactly that, not the
        // ten characters the previous call already spent.
        validator.validate_str("<a>plain text, no entities</a>")?;
        assert_eq!(validator.budget().expansions(), 0);
        assert_eq!(validator.budget().expanded_size(), 0);
        Ok(())
    }

    #[test]
    fn the_document_element_must_match_the_doctype_name() -> Result<(), Error> {
        let report = validate_str(r#"<!DOCTYPE a [<!ELEMENT b EMPTY>]><b/>"#)?;
        assert!(
            report
                .errors()
                .iter()
                .any(|error| matches!(error.kind(), ValidityErrorKind::RootElementType { .. }))
        );
        Ok(())
    }

    #[test]
    fn internal_entities_are_expanded_as_content() -> Result<(), Error> {
        let report = validate_str(
            r#"<!DOCTYPE a [
                 <!ELEMENT a (b)>
                 <!ELEMENT b (#PCDATA)>
                 <!ENTITY inner "<b>text</b>">
               ]><a>&inner;</a>"#,
        )?;
        assert!(report.is_valid(), "{:?}", report.errors());
        Ok(())
    }

    #[test]
    fn entity_content_is_validated_too() -> Result<(), Error> {
        let report = validate_str(
            r#"<!DOCTYPE a [
                 <!ELEMENT a (b)>
                 <!ELEMENT b EMPTY>
                 <!ENTITY inner "<c/>">
                 <!ELEMENT c EMPTY>
               ]><a>&inner;</a>"#,
        )?;
        assert!(!report.is_valid());
        Ok(())
    }

    #[test]
    fn character_references_become_character_data() -> Result<(), Error> {
        let report =
            validate_str(r#"<!DOCTYPE a [<!ELEMENT a (#PCDATA)>]><a>&#65;&#x42;&amp;</a>"#)?;
        assert!(report.is_valid());
        Ok(())
    }

    #[test]
    fn attribute_defaults_from_the_external_subset_break_standalone() -> Result<(), Error> {
        let resolver =
            MemoryResolver::new().with("ext.dtd", r#"<!ELEMENT a EMPTY><!ATTLIST a d CDATA "v">"#);
        let report = DocumentValidator::new()
            .with_resolver(Box::new(resolver))
            .validate_str(
                r#"<?xml version="1.0" standalone="yes"?><!DOCTYPE a SYSTEM "ext.dtd"><a/>"#,
            )?;
        assert_eq!(report.standalone(), Some(true));
        assert!(
            report
                .errors()
                .iter()
                .any(|error| matches!(error.kind(), ValidityErrorKind::StandaloneViolation { .. }))
        );
        Ok(())
    }

    #[test]
    fn documents_without_a_doctype_validate_vacuously() -> Result<(), Error> {
        let report = validate_str("<a><b/></a>")?;
        assert!(
            report
                .errors()
                .iter()
                .all(|error| matches!(error.kind(), ValidityErrorKind::ElementNotDeclared(_)))
        );
        Ok(())
    }
}
