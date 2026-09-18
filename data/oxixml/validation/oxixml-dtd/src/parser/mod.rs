//! Reading a document type definition into a [`Dtd`].
//!
//! [`DtdParser`] is the entry point. It carries the two policy knobs that make
//! DTD processing safe on untrusted input — the
//! [`EntityResolver`] (which is [`DenyExternalEntities`]
//! by default, so nothing external is fetched) and the
//! [`ExpansionLimits`] — and offers three ways in:
//!
//! * [`parse_doctype`](DtdParser::parse_doctype) — the whole document type
//!   declaration, `name ExternalID? ('[' … ']')?`, exactly the payload of the
//!   [`Event::DocType`](oxixml_xml::events::Event::DocType) event; the internal
//!   subset is read first and the external subset (when the resolver supplies
//!   it) second, so that internal declarations take precedence as XML 1.0 §2.8
//!   requires.
//! * [`parse_internal_subset`](DtdParser::parse_internal_subset) — just the
//!   declarations between the square brackets.
//! * [`parse_external_subset`](DtdParser::parse_external_subset) — a standalone
//!   `.dtd` file, where conditional sections and parameter entities inside
//!   declarations are permitted.

mod content;

pub use content::{MAX_CONTENT_MODEL_DEPTH, MAX_CONTENT_MODEL_POSITIONS};
mod scanner;
mod subset;

use crate::error::{DtdError, DtdErrorKind, ValidityError, ValidityErrorKind};
use crate::limits::{Budget, ExpansionLimits};
use crate::model::{DefaultDecl, Dtd};
use crate::names::normalize_line_ends;
use crate::resolver::{DenyExternalEntities, EntityResolver, ExternalKind};
use scanner::Scanner;
use subset::{Stop, SubsetParser};

/// Reads DTD text into a [`Dtd`].
///
/// # Examples
///
/// ```
/// use oxixml_dtd::{DtdParser, ExpansionLimits};
///
/// let dtd = DtdParser::new()
///     .with_limits(ExpansionLimits::STRICT)
///     .parse_doctype(r#"greeting SYSTEM "unused.dtd" [ <!ELEMENT greeting (#PCDATA)> ]"#);
///
/// // The external subset was named but the default resolver refuses it, so the
/// // parse fails rather than silently fetching `unused.dtd`.
/// assert!(dtd.is_err());
///
/// let dtd = DtdParser::new().parse_doctype("greeting [ <!ELEMENT greeting (#PCDATA)> ]")?;
/// assert_eq!(dtd.name(), Some("greeting"));
/// # Ok::<(), oxixml_dtd::DtdError>(())
/// ```
pub struct DtdParser {
    resolver: Box<dyn EntityResolver>,
    limits: ExpansionLimits,
    budget: Budget,
}

impl Default for DtdParser {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for DtdParser {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DtdParser")
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl DtdParser {
    /// Creates a parser that refuses every external entity and uses
    /// [`ExpansionLimits::DEFAULT`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            resolver: Box::new(DenyExternalEntities),
            limits: ExpansionLimits::DEFAULT,
            budget: ExpansionLimits::DEFAULT.budget(),
        }
    }

    /// Installs an [`EntityResolver`], enabling external entities.
    #[must_use]
    pub fn with_resolver(mut self, resolver: Box<dyn EntityResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    /// Sets the entity-expansion budget.
    #[must_use]
    pub fn with_limits(mut self, limits: ExpansionLimits) -> Self {
        self.limits = limits;
        self.budget = limits.budget();
        self
    }

    /// The configured limits.
    #[must_use]
    pub fn limits(&self) -> ExpansionLimits {
        self.limits
    }

    /// How much of the expansion budget the last parse consumed.
    ///
    /// [`DocumentValidator`](crate::DocumentValidator) shares this same running total with the
    /// content it validates after the DTD is read — it seeds its own expansion budget from this
    /// value and writes the further cost of validating a document's content back into it once
    /// finished — so a parser driven through [`DocumentValidator`](crate::DocumentValidator)
    /// reports the combined DTD-and-content total here, not just the DTD-phase share.
    #[must_use]
    pub fn budget(&self) -> Budget {
        self.budget
    }

    /// Overwrites the expansion budget, keeping the configured [`limits`](Self::limits).
    ///
    /// Used by [`DocumentValidator`](crate::DocumentValidator) to write the cost of validating a
    /// document's content back into this parser's own running total, so that the two phases
    /// share one genuinely combined budget rather than each silently getting its own full
    /// allowance.
    pub(crate) fn set_budget(&mut self, budget: Budget) {
        self.budget = budget;
    }

    /// The installed resolver, for callers that must fetch external entities
    /// themselves — the document driver of [`crate::document`] does this to
    /// expand external parsed general entities in element content.
    pub fn resolver_mut(&mut self) -> &mut dyn EntityResolver {
        &mut *self.resolver
    }

    /// Parses the body of a document type declaration — everything between
    /// `<!DOCTYPE` and the matching `>`.
    ///
    /// # Errors
    ///
    /// Any [`DtdError`]: a syntax error, an undeclared parameter entity, a
    /// denied external entity, or an exhausted expansion budget.
    pub fn parse_doctype(&mut self, declaration: &str) -> Result<Dtd, DtdError> {
        self.parse_doctype_with_base(declaration, None)
    }

    /// [`parse_doctype`](Self::parse_doctype), with the document's base URI so
    /// that relative system identifiers can be resolved.
    ///
    /// # Errors
    ///
    /// As [`parse_doctype`](Self::parse_doctype).
    pub fn parse_doctype_with_base(
        &mut self,
        declaration: &str,
        base: Option<&str>,
    ) -> Result<Dtd, DtdError> {
        let mut dtd = Dtd::new();
        self.budget = self.limits.budget();
        let text = normalize_line_ends(declaration);
        let mut external_id = None;

        {
            let mut parser = SubsetParser {
                scanner: Scanner::new(text, base.map(str::to_owned)),
                dtd: &mut dtd,
                resolver: &mut *self.resolver,
                budget: &mut self.budget,
                internal: true,
                group_nesting_error: false,
            };
            parser.skip_whitespace(false)?;
            let name = parser.read_name()?;
            parser.dtd.set_name(name);
            parser.skip_whitespace(false)?;
            if matches!(parser.scanner.peek(), Some('S' | 'P')) {
                let id = parser.read_external_id(false)?;
                parser.dtd.set_external_id(id.clone());
                external_id = Some(id);
                parser.skip_whitespace(false)?;
            }
            if parser.scanner.peek() == Some('[') {
                parser.scanner.next_char();
                parser.run(Stop::CloseBracket)?;
                parser.expect_char(']')?;
            }
            parser.skip_whitespace(false)?;
            if !parser.scanner.is_eof() {
                let found = parser.describe_next();
                return Err(parser.error(DtdErrorKind::Unexpected {
                    expected: "the end of the document type declaration".to_owned(),
                    found,
                }));
            }
        }

        if let Some(id) = external_id {
            let system_id = id.system_id().unwrap_or_default().to_owned();
            let public_id = id.public_id().map(str::to_owned);
            let (text, resolved_base) = {
                let mut parser = SubsetParser {
                    scanner: Scanner::new(String::new(), base.map(str::to_owned)),
                    dtd: &mut dtd,
                    resolver: &mut *self.resolver,
                    budget: &mut self.budget,
                    internal: false,
                    group_nesting_error: false,
                };
                parser.fetch_external(
                    ExternalKind::Subset,
                    None,
                    public_id.as_deref(),
                    &system_id,
                    base,
                )?
            };
            self.parse_subset_into(&mut dtd, &text, false, resolved_base.as_deref())?;
            dtd.set_external_subset_read(true);
        }

        self.finish(&mut dtd)?;
        Ok(dtd)
    }

    /// Parses the declarations of an internal subset — the text between the
    /// square brackets of a document type declaration.
    ///
    /// # Errors
    ///
    /// As [`parse_doctype`](Self::parse_doctype).
    pub fn parse_internal_subset(&mut self, text: &str) -> Result<Dtd, DtdError> {
        let mut dtd = Dtd::new();
        self.budget = self.limits.budget();
        self.parse_subset_into(&mut dtd, text, true, None)?;
        self.finish(&mut dtd)?;
        Ok(dtd)
    }

    /// Parses an external subset: a standalone DTD, where conditional sections
    /// and parameter entities inside markup declarations are permitted.
    ///
    /// # Errors
    ///
    /// As [`parse_doctype`](Self::parse_doctype).
    pub fn parse_external_subset(&mut self, text: &str) -> Result<Dtd, DtdError> {
        self.parse_external_subset_with_base(text, None)
    }

    /// [`parse_external_subset`](Self::parse_external_subset), with a base URI
    /// for relative system identifiers.
    ///
    /// # Errors
    ///
    /// As [`parse_doctype`](Self::parse_doctype).
    pub fn parse_external_subset_with_base(
        &mut self,
        text: &str,
        base: Option<&str>,
    ) -> Result<Dtd, DtdError> {
        let mut dtd = Dtd::new();
        self.budget = self.limits.budget();
        self.parse_subset_into(&mut dtd, text, false, base)?;
        dtd.set_external_subset_read(true);
        self.finish(&mut dtd)?;
        Ok(dtd)
    }

    /// Reads a subset's declarations into an existing [`Dtd`].
    fn parse_subset_into(
        &mut self,
        dtd: &mut Dtd,
        text: &str,
        internal: bool,
        base: Option<&str>,
    ) -> Result<(), DtdError> {
        let normalized = normalize_line_ends(text);
        let body = if internal {
            normalized.as_str()
        } else {
            strip_text_declaration(&normalized)
        };
        let mut parser = SubsetParser {
            scanner: Scanner::new(body.to_owned(), base.map(str::to_owned)),
            dtd,
            resolver: &mut *self.resolver,
            budget: &mut self.budget,
            internal,
            group_nesting_error: false,
        };
        parser.run(Stop::Eof)
    }

    /// Normalizes attribute defaults and runs the DTD-level validity checks.
    fn finish(&mut self, dtd: &mut Dtd) -> Result<(), DtdError> {
        normalize_attribute_defaults(dtd, &mut self.budget)?;
        dtd.check_attribute_declarations();
        check_default_values(dtd);
        Ok(())
    }
}

/// Rewrites every attribute default into its normalized form (XML 1.0 §3.3.3),
/// now that every entity declaration is known.
///
/// A default that references an entity the DTD does not declare, or an external
/// one, is left as written: that is a well-formedness question for the document
/// parser, not a reason to reject the DTD. Recursion and expansion-budget
/// failures *are* propagated — they are exactly what the guard exists for.
fn normalize_attribute_defaults(dtd: &mut Dtd, budget: &mut Budget) -> Result<(), DtdError> {
    let mut updates: Vec<(String, String, DefaultDecl)> = Vec::new();
    for list in dtd.attribute_lists() {
        for declaration in list.iter() {
            let tokenized = declaration.att_type().is_tokenized();
            let raw = match declaration.default() {
                DefaultDecl::Fixed(value) | DefaultDecl::Default(value) => value,
                DefaultDecl::Required | DefaultDecl::Implied => continue,
            };
            let normalized = match dtd.normalize_attribute_value(raw, tokenized, budget) {
                Ok(value) => value,
                Err(error) => match error.kind() {
                    DtdErrorKind::UndeclaredGeneralEntity(_)
                    | DtdErrorKind::ExternalEntityDenied { .. }
                    | DtdErrorKind::ExternalEntityUnavailable { .. } => continue,
                    _ => return Err(error),
                },
            };
            let replacement = match declaration.default() {
                DefaultDecl::Fixed(_) => DefaultDecl::Fixed(normalized),
                _ => DefaultDecl::Default(normalized),
            };
            updates.push((
                list.element().to_owned(),
                declaration.name().to_owned(),
                replacement,
            ));
        }
    }
    for (element, attribute, default) in updates {
        if let Some(list) = dtd.attributes_mut(&element)
            && let Some(declaration) = list.get_mut(&attribute)
        {
            declaration.set_default(default);
        }
    }
    Ok(())
}

/// VC: Attribute Default Value Syntactically Correct (XML 1.0 §3.3.2).
fn check_default_values(dtd: &mut Dtd) {
    let mut errors = Vec::new();
    for list in dtd.attribute_lists() {
        for declaration in list.iter() {
            let Some(value) = declaration.default().value() else {
                continue;
            };
            if let Some(reason) =
                crate::validate::attributes::syntactic_error(declaration.att_type(), value)
            {
                errors.push(ValidityError::detached(
                    ValidityErrorKind::DefaultValueInvalid {
                        element: list.element().to_owned(),
                        attribute: declaration.name().to_owned(),
                        value: value.to_owned(),
                        reason,
                    },
                ));
            }
        }
    }
    for error in errors {
        dtd.push_declaration_error(error);
    }
}

/// Removes a leading `<?xml … ?>` text declaration from an external entity.
///
/// XML 1.0 §4.3.1 allows — and for entities in a non-UTF encoding requires —
/// external parsed entities to begin with one. It is not a processing
/// instruction and must not be parsed as one.
pub(crate) fn strip_text_declaration(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("<?xml") else {
        return text;
    };
    // `<?xml-stylesheet …?>` is an ordinary processing instruction.
    if !matches!(rest.chars().next(), Some(' ' | '\t' | '\r' | '\n')) {
        return text;
    }
    match rest.find("?>") {
        Some(end) => &rest[end + 2..],
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AttributeType, EntityDecl};
    use crate::resolver::MemoryResolver;

    #[test]
    fn text_declarations_are_stripped_only_when_leading() {
        assert_eq!(
            strip_text_declaration("<?xml version=\"1.0\"?>rest"),
            "rest"
        );
        assert_eq!(
            strip_text_declaration("<?xml-stylesheet?>rest"),
            "<?xml-stylesheet?>rest"
        );
        assert_eq!(strip_text_declaration("rest"), "rest");
    }

    #[test]
    fn doctype_declarations_yield_the_root_name_and_internal_subset() -> Result<(), crate::DtdError>
    {
        let dtd = DtdParser::new().parse_doctype("root [ <!ELEMENT root EMPTY> ]")?;
        assert_eq!(dtd.name(), Some("root"));
        assert!(dtd.element("root").is_some());
        assert!(!dtd.has_external_subset());
        Ok(())
    }

    #[test]
    fn the_internal_subset_wins_over_the_external_one() -> Result<(), crate::DtdError> {
        let resolver = MemoryResolver::new().with("ext.dtd", r#"<!ENTITY who "external">"#);
        let dtd = DtdParser::new()
            .with_resolver(Box::new(resolver))
            .parse_doctype(r#"root SYSTEM "ext.dtd" [ <!ENTITY who "internal"> ]"#)?;
        assert_eq!(
            dtd.general_entity("who")
                .and_then(EntityDecl::internal_text),
            Some("internal")
        );
        assert!(dtd.external_subset_read());
        Ok(())
    }

    #[test]
    fn attribute_defaults_are_normalized_after_the_whole_dtd_is_read() -> Result<(), crate::DtdError>
    {
        let dtd = DtdParser::new().parse_internal_subset(
            r#"<!ATTLIST a x NMTOKENS " &later;  b "><!ENTITY later "z">"#,
        )?;
        let value = dtd
            .attributes("a")
            .and_then(|list| list.get("x"))
            .and_then(|declaration| declaration.default().value())
            .map(str::to_owned);
        assert_eq!(value.as_deref(), Some("z b"));
        Ok(())
    }

    #[test]
    fn notation_attribute_types_parse() -> Result<(), crate::DtdError> {
        let dtd = DtdParser::new().parse_internal_subset(
            r#"<!NOTATION gif SYSTEM "g"><!ATTLIST a t NOTATION (gif) #IMPLIED>"#,
        )?;
        let att_type = dtd
            .attributes("a")
            .and_then(|list| list.get("t"))
            .map(|declaration| declaration.att_type().clone());
        assert_eq!(
            att_type,
            Some(AttributeType::Notation(vec!["gif".to_owned()]))
        );
        Ok(())
    }
}
