//! The recursive-descent parser for the DTD subsets.
//!
//! Implements the `markupdecl` production of XML 1.0 §2.8 `[29]` and everything
//! it reaches: element type declarations, attribute-list declarations, entity
//! declarations, notation declarations, processing instructions, comments and
//! (in the external subset) conditional sections.

use crate::error::{DtdError, DtdErrorKind, ValidityError, ValidityErrorKind};
use crate::limits::Budget;
use crate::model::{
    AttributeDecl, AttributeType, ContentSpec, DefaultDecl, Dtd, ElementDecl, EntityDecl,
    EntityDefinition, EntityKind, ExternalId, NotationDecl,
};
use crate::names::{is_name_char, is_name_start_char, is_pubid_char, is_whitespace};
use crate::parser::content::MAX_CONTENT_MODEL_POSITIONS;
use crate::parser::scanner::Scanner;
use crate::resolver::{EntityResolver, ExternalKind, ResolveRequest, ResolverError};

/// What ends a run of markup declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stop {
    /// End of the input text.
    Eof,
    /// The `]` that closes an internal subset. Not consumed.
    CloseBracket,
    /// The `]]>` that closes an `INCLUDE` conditional section, carrying the
    /// frame the section's `<![` came from so that VC: Proper Conditional
    /// Section/PE Nesting can be checked. Consumed.
    ConditionalEnd(u64),
}

/// One pass over a DTD subset.
pub(crate) struct SubsetParser<'a> {
    pub(crate) scanner: Scanner,
    pub(crate) dtd: &'a mut Dtd,
    pub(crate) resolver: &'a mut dyn EntityResolver,
    pub(crate) budget: &'a mut Budget,
    /// Whether the *base* frame is the internal subset, where the "PEs in
    /// Internal Subset" well-formedness constraint applies.
    pub(crate) internal: bool,
    /// Set when a parenthesised group of the content model currently being read
    /// opened in one entity and closed in another, violating VC: Proper
    /// Group/PE Nesting (XML 1.0 §3.2.1).
    pub(crate) group_nesting_error: bool,
}

impl SubsetParser<'_> {
    /// Builds an error at the current position.
    pub(crate) fn error(&self, kind: DtdErrorKind) -> DtdError {
        DtdError::new(kind, self.scanner.position(), self.scanner.entity_stack())
    }

    /// Attaches position information to an error raised without any.
    pub(crate) fn locate(&self, error: DtdError) -> DtdError {
        if error.position() == oxixml_support::pos::TextPosition::start()
            && error.entity_stack().is_empty()
        {
            return self.error(error.into_kind());
        }
        error
    }

    /// Whether declarations read here count as coming from outside the internal
    /// subset, for the standalone-document constraint of XML 1.0 §2.9.
    pub(crate) fn is_external_decl(&mut self) -> bool {
        !self.internal || !self.scanner.in_base_text()
    }

    /// Reads markup declarations until `stop`.
    pub(crate) fn run(&mut self, stop: Stop) -> Result<(), DtdError> {
        loop {
            self.skip_whitespace_and_parameter_entities()?;
            match stop {
                Stop::ConditionalEnd(open_frame) => {
                    if self.scanner.try_consume("]]>") {
                        self.check_conditional_nesting(open_frame);
                        return Ok(());
                    }
                }
                Stop::CloseBracket => {
                    if self.scanner.peek() == Some(']') {
                        return Ok(());
                    }
                }
                Stop::Eof => {}
            }
            if self.scanner.is_eof() {
                return match stop {
                    Stop::Eof => Ok(()),
                    Stop::CloseBracket => Err(self.error(DtdErrorKind::UnexpectedEof {
                        expected: "`]`".to_owned(),
                    })),
                    Stop::ConditionalEnd(_) => Err(self.error(DtdErrorKind::UnexpectedEof {
                        expected: "`]]>`".to_owned(),
                    })),
                };
            }
            self.parse_markup_decl()?;
        }
    }

    /// Dispatches on the kind of markup declaration that comes next.
    fn parse_markup_decl(&mut self) -> Result<(), DtdError> {
        let opening_frame = self.scanner.current_frame_id();
        if self.scanner.try_consume("<!--") {
            return self.skip_comment();
        }
        if self.scanner.try_consume("<?") {
            return self.skip_processing_instruction();
        }
        if self.scanner.try_consume("<![") {
            let open_frame = self.scanner.last_frame_id();
            return self.parse_conditional_section(open_frame);
        }
        if self.scanner.try_consume("<!ELEMENT") {
            self.parse_element_decl()?;
        } else if self.scanner.try_consume("<!ATTLIST") {
            self.parse_attlist_decl()?;
        } else if self.scanner.try_consume("<!ENTITY") {
            self.parse_entity_decl()?;
        } else if self.scanner.try_consume("<!NOTATION") {
            self.parse_notation_decl()?;
        } else {
            let found = self.describe_next();
            return Err(self.error(DtdErrorKind::Unexpected {
                expected: "a markup declaration".to_owned(),
                found,
            }));
        }
        self.check_declaration_nesting(opening_frame)
    }

    /// VC: Proper Declaration/PE Nesting (XML 1.0 §2.8) — a markup declaration
    /// must start and finish in the same entity.
    ///
    /// This is a validity error rather than a fatal one, so parsing continues
    /// and the violation is recorded on the DTD.
    fn check_declaration_nesting(&mut self, opening_frame: u64) -> Result<(), DtdError> {
        if self.scanner.last_frame_id() == opening_frame {
            return Ok(());
        }
        let entity = self.scanner.current_entity().unwrap_or_default().to_owned();
        self.dtd.push_declaration_error(ValidityError::detached(
            ValidityErrorKind::ImproperDeclarationNesting { entity },
        ));
        Ok(())
    }

    /// A short rendering of what comes next, for error messages.
    pub(crate) fn describe_next(&mut self) -> String {
        match self.scanner.peek() {
            Some(c) => format!("`{c}`"),
            None => "end of input".to_owned(),
        }
    }

    // ---------------------------------------------------------------- tokens

    /// Consumes white space; expands any parameter-entity reference it meets.
    ///
    /// `inside_declaration` selects the "PEs in Internal Subset" check: inside a
    /// markup declaration in the internal subset, a parameter-entity reference
    /// is a well-formedness error.
    pub(crate) fn skip_whitespace(&mut self, inside_declaration: bool) -> Result<(), DtdError> {
        loop {
            self.scanner.take_while(is_whitespace);
            if self.scanner.peek() != Some('%') {
                return Ok(());
            }
            if !self.looks_like_parameter_entity_reference() {
                return Ok(());
            }
            if inside_declaration && self.internal && self.scanner.in_base_text() {
                let name = self.preview_parameter_entity_name();
                return Err(self.error(DtdErrorKind::ParameterEntityInInternalSubset(name)));
            }
            self.expand_parameter_entity()?;
        }
    }

    /// Consumes white space between markup declarations, expanding parameter
    /// entities, which is always allowed there.
    fn skip_whitespace_and_parameter_entities(&mut self) -> Result<(), DtdError> {
        loop {
            self.scanner.take_while(is_whitespace);
            if self.scanner.peek() != Some('%') || !self.looks_like_parameter_entity_reference() {
                return Ok(());
            }
            self.expand_parameter_entity()?;
        }
    }

    /// Requires at least one white-space character (or a parameter entity that
    /// supplies the separation).
    pub(crate) fn expect_whitespace(&mut self) -> Result<(), DtdError> {
        let separated = matches!(self.scanner.peek(), Some(c) if is_whitespace(c))
            || self.scanner.peek() == Some('%');
        if !separated {
            let found = self.describe_next();
            return Err(self.error(DtdErrorKind::Unexpected {
                expected: "white space".to_owned(),
                found,
            }));
        }
        self.skip_whitespace(true)
    }

    /// Whether `%` really starts a parameter-entity reference rather than being
    /// a stray character.
    fn looks_like_parameter_entity_reference(&mut self) -> bool {
        matches!(self.scanner.peek_ahead(1), Some(c) if is_name_start_char(c))
    }

    /// The name in the parameter-entity reference that comes next, for use in
    /// an error message.
    fn preview_parameter_entity_name(&mut self) -> String {
        self.scanner.rest()[1..]
            .chars()
            .take_while(|&c| is_name_char(c))
            .collect()
    }

    /// Consumes `expected`, or fails.
    pub(crate) fn expect_char(&mut self, expected: char) -> Result<(), DtdError> {
        if self.scanner.peek() == Some(expected) {
            self.scanner.next_char();
            return Ok(());
        }
        let found = self.describe_next();
        Err(self.error(DtdErrorKind::Unexpected {
            expected: format!("`{expected}`"),
            found,
        }))
    }

    /// Reads a `Name`.
    pub(crate) fn read_name(&mut self) -> Result<String, DtdError> {
        let name = self.scanner.take_while(is_name_char);
        if name.is_empty() || !oxixml_qname::is_name(&name) {
            return Err(self.error(DtdErrorKind::InvalidName(name)));
        }
        Ok(name)
    }

    /// Reads an `Nmtoken`.
    pub(crate) fn read_nmtoken(&mut self) -> Result<String, DtdError> {
        let token = self.scanner.take_while(is_name_char);
        if token.is_empty() || !oxixml_qname::is_nmtoken(&token) {
            return Err(self.error(DtdErrorKind::InvalidNmtoken(token)));
        }
        Ok(token)
    }

    /// Reads a `SystemLiteral` (XML 1.0 §4.2.2 `[11]`).
    pub(crate) fn read_system_literal(&mut self) -> Result<String, DtdError> {
        let quote = self.read_quote()?;
        let frame = self.scanner.current_frame_id();
        let mut out = String::new();
        loop {
            let Some(c) = self.scanner.next_char() else {
                return Err(self.error(DtdErrorKind::UnexpectedEof {
                    expected: format!("`{quote}`"),
                }));
            };
            if c == quote && self.scanner.last_frame_id() == frame {
                return Ok(out);
            }
            out.push(c);
        }
    }

    /// Reads a `PubidLiteral` (XML 1.0 §4.2.2 `[12]`), collapsing white space as
    /// §4.2.2 requires.
    pub(crate) fn read_pubid_literal(&mut self) -> Result<String, DtdError> {
        let quote = self.read_quote()?;
        let frame = self.scanner.current_frame_id();
        let mut out = String::new();
        loop {
            let Some(c) = self.scanner.next_char() else {
                return Err(self.error(DtdErrorKind::UnexpectedEof {
                    expected: format!("`{quote}`"),
                }));
            };
            if c == quote && self.scanner.last_frame_id() == frame {
                break;
            }
            if !is_pubid_char(c) || (quote == '\'' && c == '\'') {
                return Err(self.error(DtdErrorKind::InvalidPublicId(out)));
            }
            out.push(c);
        }
        let collapsed: Vec<&str> = out.split_whitespace().collect();
        Ok(collapsed.join(" "))
    }

    /// Consumes an opening quote character.
    fn read_quote(&mut self) -> Result<char, DtdError> {
        match self.scanner.peek() {
            Some(quote @ ('"' | '\'')) => {
                self.scanner.next_char();
                Ok(quote)
            }
            _ => {
                let found = self.describe_next();
                Err(self.error(DtdErrorKind::Unexpected {
                    expected: "a quoted literal".to_owned(),
                    found,
                }))
            }
        }
    }

    /// Reads an `AttValue` without normalizing it: general-entity references are
    /// left in place for [`Dtd::normalize_attribute_value`], which runs once the
    /// whole DTD is known.
    pub(crate) fn read_attribute_value(&mut self) -> Result<String, DtdError> {
        let quote = self.read_quote()?;
        let frame = self.scanner.current_frame_id();
        let mut out = String::new();
        loop {
            let Some(c) = self.scanner.next_char() else {
                return Err(self.error(DtdErrorKind::UnexpectedEof {
                    expected: format!("`{quote}`"),
                }));
            };
            if c == quote && self.scanner.last_frame_id() == frame {
                return Ok(out);
            }
            if c == '<' {
                return Err(self.error(DtdErrorKind::Unexpected {
                    expected: "an attribute value".to_owned(),
                    found: "`<`".to_owned(),
                }));
            }
            out.push(c);
        }
    }

    /// Reads an `EntityValue` (XML 1.0 §4.2 `[9]`).
    ///
    /// Applies the three inclusion rules of §4.4 that hold inside a literal:
    /// parameter-entity references are *included in literal* (expanded here and
    /// now, without the surrounding spaces), character references are
    /// *included* (resolved to the character), and general-entity references are
    /// *bypassed* (left spelled out, to be expanded when the entity is used).
    pub(crate) fn read_entity_value(&mut self) -> Result<String, DtdError> {
        let quote = self.read_quote()?;
        let frame = self.scanner.current_frame_id();
        let mut out = String::new();
        loop {
            let Some(c) = self.scanner.peek() else {
                return Err(self.error(DtdErrorKind::UnexpectedEof {
                    expected: format!("`{quote}`"),
                }));
            };
            if c == '%' && self.looks_like_parameter_entity_reference() {
                self.expand_parameter_entity_in_literal()?;
                continue;
            }
            self.scanner.next_char();
            if c == quote && self.scanner.last_frame_id() == frame {
                return Ok(out);
            }
            if c != '&' {
                out.push(c);
                continue;
            }
            if self.scanner.peek() == Some('#') {
                self.scanner.next_char();
                let body = self.scanner.take_while(|c| c != ';' && !is_whitespace(c));
                self.expect_char(';')?;
                let Some(resolved) = crate::entity::resolve_char_ref(&body) else {
                    return Err(self.error(DtdErrorKind::InvalidCharRef(body)));
                };
                out.push(resolved);
            } else {
                let name = self.read_name()?;
                self.expect_char(';')?;
                out.push('&');
                out.push_str(&name);
                out.push(';');
            }
        }
    }

    // -------------------------------------------------------------- entities

    /// Expands the parameter-entity reference at the cursor, padding the
    /// replacement text with spaces as XML 1.0 §4.4.8 requires in the DTD.
    fn expand_parameter_entity(&mut self) -> Result<(), DtdError> {
        self.expand_parameter_entity_with_padding(true)
    }

    /// Expands the parameter-entity reference at the cursor *inside a literal*,
    /// where §4.4.5 says the replacement text is included without padding.
    fn expand_parameter_entity_in_literal(&mut self) -> Result<(), DtdError> {
        self.expand_parameter_entity_with_padding(false)
    }

    /// Shared implementation of parameter-entity inclusion.
    fn expand_parameter_entity_with_padding(&mut self, padded: bool) -> Result<(), DtdError> {
        self.expect_char('%')?;
        let name = self.read_name()?;
        self.expect_char(';')?;
        self.dtd.note_parameter_entity_reference();

        if self
            .scanner
            .entity_stack()
            .iter()
            .any(|entry| entry == &name)
        {
            return Err(self.error(DtdErrorKind::RecursiveEntity {
                name,
                chain: self.scanner.entity_stack(),
            }));
        }
        let Some(entity) = self.dtd.parameter_entity(&name) else {
            return Err(self.error(DtdErrorKind::UndeclaredParameterEntity(name)));
        };
        let declared_base = entity.base().map(str::to_owned);
        let text = match entity.definition() {
            EntityDefinition::Internal(text) => text.clone(),
            EntityDefinition::External { id, .. } => {
                let system_id = id.system_id().unwrap_or_default().to_owned();
                let public_id = id.public_id().map(str::to_owned);
                let resolved = self.fetch_external(
                    ExternalKind::ParameterEntity,
                    Some(&name),
                    public_id.as_deref(),
                    &system_id,
                    declared_base.as_deref(),
                )?;
                self.budget
                    .check_depth(self.scanner.entity_depth() + 1)
                    .map_err(|e| self.locate(e))?;
                self.budget
                    .charge_expansion(resolved.0.chars().count())
                    .map_err(|e| self.locate(e))?;
                self.scanner
                    .push_entity(name, &resolved.0, resolved.1, padded);
                return Ok(());
            }
        };
        self.budget
            .check_depth(self.scanner.entity_depth() + 1)
            .map_err(|e| self.locate(e))?;
        self.budget
            .charge_expansion(text.chars().count())
            .map_err(|e| self.locate(e))?;
        self.scanner.push_entity(name, &text, declared_base, padded);
        Ok(())
    }

    /// Asks the resolver for an external entity, returning its text and base URI.
    pub(crate) fn fetch_external(
        &mut self,
        kind: ExternalKind,
        name: Option<&str>,
        public_id: Option<&str>,
        system_id: &str,
        declared_base: Option<&str>,
    ) -> Result<(String, Option<String>), DtdError> {
        let base = declared_base.or_else(|| self.scanner.base());
        let request = ResolveRequest {
            kind,
            name,
            public_id,
            system_id,
            base,
        };
        match self.resolver.resolve(&request) {
            Ok(resolved) => {
                let text = crate::names::normalize_line_ends(&resolved.text);
                let body = crate::parser::strip_text_declaration(&text).to_owned();
                Ok((body, resolved.base))
            }
            Err(ResolverError::Denied) => Err(self.error(DtdErrorKind::ExternalEntityDenied {
                system_id: system_id.to_owned(),
            })),
            Err(ResolverError::Unavailable(reason)) => {
                Err(self.error(DtdErrorKind::ExternalEntityUnavailable {
                    system_id: system_id.to_owned(),
                    reason,
                }))
            }
        }
    }

    // ---------------------------------------------------------- declarations

    /// Skips the rest of a comment, enforcing the `--` restriction of §2.5.
    fn skip_comment(&mut self) -> Result<(), DtdError> {
        match self.scanner.take_until("-->") {
            Some(body) => {
                if body.contains("--") {
                    return Err(self.error(DtdErrorKind::Unexpected {
                        expected: "`-->`".to_owned(),
                        found: "`--` inside a comment".to_owned(),
                    }));
                }
                Ok(())
            }
            None => Err(self.error(DtdErrorKind::UnexpectedEof {
                expected: "`-->`".to_owned(),
            })),
        }
    }

    /// Skips the rest of a processing instruction.
    fn skip_processing_instruction(&mut self) -> Result<(), DtdError> {
        let target = self.read_name()?;
        if target.eq_ignore_ascii_case("xml") && !self.scanner.rest().starts_with("?>") {
            // An `<?xml …?>` text declaration is only legal at the very start of
            // an external entity; the parser strips it before we get here.
            return Err(self.error(DtdErrorKind::Unexpected {
                expected: "a processing-instruction target other than `xml`".to_owned(),
                found: format!("`{target}`"),
            }));
        }
        if self.scanner.take_until("?>").is_none() {
            return Err(self.error(DtdErrorKind::UnexpectedEof {
                expected: "`?>`".to_owned(),
            }));
        }
        Ok(())
    }

    /// Parses `INCLUDE` / `IGNORE` conditional sections (XML 1.0 §3.4 `[61]`).
    fn parse_conditional_section(&mut self, open_frame: u64) -> Result<(), DtdError> {
        if self.internal && self.scanner.in_base_text() {
            return Err(self.error(DtdErrorKind::ConditionalSectionInInternalSubset));
        }
        self.skip_whitespace(true)?;
        if self.scanner.try_consume("INCLUDE") {
            self.skip_whitespace(true)?;
            self.expect_char('[')?;
            self.check_conditional_nesting(open_frame);
            return self.run(Stop::ConditionalEnd(open_frame));
        }
        if self.scanner.try_consume("IGNORE") {
            self.skip_whitespace(true)?;
            self.expect_char('[')?;
            self.check_conditional_nesting(open_frame);
            return self.skip_ignored_section(open_frame);
        }
        let keyword = self.scanner.take_while(is_name_char);
        Err(self.error(DtdErrorKind::BadConditionalSection(keyword)))
    }

    /// VC: Proper Conditional Section/PE Nesting (XML 1.0 §3.4) — every one of
    /// `<![`, `[` and `]]>` must come from the same entity.
    fn check_conditional_nesting(&mut self, open_frame: u64) {
        if self.scanner.last_frame_id() != open_frame {
            self.dtd.push_declaration_error(ValidityError::detached(
                ValidityErrorKind::ImproperConditionalSectionNesting,
            ));
        }
    }

    /// Skips the body of an `IGNORE` section, tracking nested sections.
    fn skip_ignored_section(&mut self, open_frame: u64) -> Result<(), DtdError> {
        let mut depth = 1_usize;
        loop {
            if self.scanner.try_consume("<![") {
                depth += 1;
            } else if self.scanner.try_consume("]]>") {
                depth -= 1;
                if depth == 0 {
                    self.check_conditional_nesting(open_frame);
                    return Ok(());
                }
            } else if self.scanner.next_char().is_none() {
                return Err(self.error(DtdErrorKind::UnexpectedEof {
                    expected: "`]]>`".to_owned(),
                }));
            }
        }
    }

    /// Parses `<!ELEMENT name contentspec>`.
    fn parse_element_decl(&mut self) -> Result<(), DtdError> {
        let external = self.is_external_decl();
        self.expect_whitespace()?;
        let name = self.read_name()?;
        self.expect_whitespace()?;
        self.group_nesting_error = false;
        let content = self.parse_content_spec()?;
        self.skip_whitespace(true)?;
        self.expect_char('>')?;
        if self.group_nesting_error {
            self.group_nesting_error = false;
            self.dtd.push_declaration_error(ValidityError::detached(
                ValidityErrorKind::ImproperGroupNesting {
                    element: name.clone(),
                },
            ));
        }
        self.dtd
            .add_element(ElementDecl::new(name, content, external));
        Ok(())
    }

    /// Parses `<!ATTLIST name AttDef*>`.
    fn parse_attlist_decl(&mut self) -> Result<(), DtdError> {
        let external = self.is_external_decl();
        self.expect_whitespace()?;
        let element = self.read_name()?;
        loop {
            self.skip_whitespace(true)?;
            if self.scanner.peek() == Some('>') {
                self.scanner.next_char();
                return Ok(());
            }
            let name = self.read_name()?;
            self.expect_whitespace()?;
            let att_type = self.parse_attribute_type()?;
            self.expect_whitespace()?;
            let default = self.parse_default_decl()?;
            self.dtd.add_attribute(
                &element,
                AttributeDecl::new(name, att_type, default, external),
            );
        }
    }

    /// Parses a `DefaultDecl` (XML 1.0 §3.3.2 `[60]`).
    fn parse_default_decl(&mut self) -> Result<DefaultDecl, DtdError> {
        if self.scanner.try_consume("#REQUIRED") {
            return Ok(DefaultDecl::Required);
        }
        if self.scanner.try_consume("#IMPLIED") {
            return Ok(DefaultDecl::Implied);
        }
        if self.scanner.try_consume("#FIXED") {
            self.expect_whitespace()?;
            return Ok(DefaultDecl::Fixed(self.read_attribute_value()?));
        }
        Ok(DefaultDecl::Default(self.read_attribute_value()?))
    }

    /// Parses `<!ENTITY …>`, general or parameter.
    fn parse_entity_decl(&mut self) -> Result<(), DtdError> {
        let external = self.is_external_decl();
        self.expect_whitespace()?;
        let kind = if self.scanner.peek() == Some('%') {
            self.scanner.next_char();
            self.expect_whitespace()?;
            EntityKind::Parameter
        } else {
            EntityKind::General
        };
        let name = self.read_name()?;
        self.expect_whitespace()?;

        let definition = if matches!(self.scanner.peek(), Some('"' | '\'')) {
            EntityDefinition::Internal(self.read_entity_value()?)
        } else {
            let id = self.read_external_id(false)?;
            let mut notation = None;
            self.skip_whitespace(true)?;
            if kind == EntityKind::General && self.scanner.try_consume("NDATA") {
                self.expect_whitespace()?;
                notation = Some(self.read_name()?);
            }
            EntityDefinition::External { id, notation }
        };
        self.skip_whitespace(true)?;
        self.expect_char('>')?;
        let base = self.scanner.base().map(str::to_owned);
        self.dtd
            .add_entity(EntityDecl::new(name, kind, definition, external).with_base(base));
        Ok(())
    }

    /// Parses `<!NOTATION name (ExternalID | PublicID)>`.
    fn parse_notation_decl(&mut self) -> Result<(), DtdError> {
        let external = self.is_external_decl();
        self.expect_whitespace()?;
        let name = self.read_name()?;
        self.expect_whitespace()?;
        let id = self.read_external_id(true)?;
        self.skip_whitespace(true)?;
        self.expect_char('>')?;
        self.dtd.add_notation(NotationDecl::new(name, id, external));
        Ok(())
    }

    /// Reads an `ExternalID`, or a bare `PublicID` when `allow_public_only`
    /// (which only a notation declaration permits).
    pub(crate) fn read_external_id(
        &mut self,
        allow_public_only: bool,
    ) -> Result<ExternalId, DtdError> {
        if self.scanner.try_consume("SYSTEM") {
            self.expect_whitespace()?;
            return Ok(ExternalId::system(self.read_system_literal()?));
        }
        if self.scanner.try_consume("PUBLIC") {
            self.expect_whitespace()?;
            let public_id = self.read_pubid_literal()?;
            if allow_public_only {
                self.skip_whitespace(true)?;
                if !matches!(self.scanner.peek(), Some('"' | '\'')) {
                    return Ok(ExternalId::public_only(public_id));
                }
            } else {
                self.expect_whitespace()?;
            }
            return Ok(ExternalId::public(public_id, self.read_system_literal()?));
        }
        let found = self.describe_next();
        Err(self.error(DtdErrorKind::Unexpected {
            expected: "`SYSTEM` or `PUBLIC`".to_owned(),
            found,
        }))
    }

    /// Parses a `contentspec` (XML 1.0 §3.2 `[46]`).
    fn parse_content_spec(&mut self) -> Result<ContentSpec, DtdError> {
        if self.scanner.try_consume("EMPTY") {
            return Ok(ContentSpec::Empty);
        }
        if self.scanner.try_consume("ANY") {
            return Ok(ContentSpec::Any);
        }
        self.expect_char('(')?;
        let open_frame = self.scanner.last_frame_id();
        self.skip_whitespace(true)?;
        if self.scanner.try_consume("#PCDATA") {
            return self.parse_mixed_tail(open_frame);
        }
        let particle = self.parse_group_tail(open_frame, 0)?;
        // The parenthesis-nesting cap above guards the parser's own call stack, but a single
        // flat, repeatable group of many distinct child names (`(a|b|…|z9999)+`) sits at
        // nesting depth 1 and is unbounded by it — yet compiling *that* shape into the
        // automaton of `crate::automaton` is superlinear in the leaf count. Reject an
        // oversized model here, before `Dtd::add_element` ever compiles it.
        if particle.leaf_count() > MAX_CONTENT_MODEL_POSITIONS {
            return Err(self.error(DtdErrorKind::ContentModelPositions(
                MAX_CONTENT_MODEL_POSITIONS,
            )));
        }
        Ok(ContentSpec::Children(particle))
    }

    /// Parses an `AttType` (XML 1.0 §3.3.1 `[54]`).
    fn parse_attribute_type(&mut self) -> Result<AttributeType, DtdError> {
        for (keyword, att_type) in [
            ("CDATA", AttributeType::Cdata),
            ("IDREFS", AttributeType::IdRefs),
            ("IDREF", AttributeType::IdRef),
            ("ID", AttributeType::Id),
            ("ENTITIES", AttributeType::Entities),
            ("ENTITY", AttributeType::Entity),
            ("NMTOKENS", AttributeType::NmTokens),
            ("NMTOKEN", AttributeType::NmToken),
        ] {
            if self.scanner.rest().starts_with(keyword)
                && !matches!(self.scanner.peek_ahead(keyword.len()), Some(c) if is_name_char(c))
            {
                self.scanner.try_consume(keyword);
                return Ok(att_type);
            }
        }
        if self.scanner.try_consume("NOTATION") {
            self.expect_whitespace()?;
            return Ok(AttributeType::Notation(self.read_token_group(true)?));
        }
        if self.scanner.peek() == Some('(') {
            return Ok(AttributeType::Enumeration(self.read_token_group(false)?));
        }
        let found = self.describe_next();
        Err(self.error(DtdErrorKind::Unexpected {
            expected: "an attribute type".to_owned(),
            found,
        }))
    }

    /// Reads `'(' S? tok (S? '|' S? tok)* S? ')'`.
    fn read_token_group(&mut self, names_only: bool) -> Result<Vec<String>, DtdError> {
        self.expect_char('(')?;
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace(true)?;
            tokens.push(if names_only {
                self.read_name()?
            } else {
                self.read_nmtoken()?
            });
            self.skip_whitespace(true)?;
            match self.scanner.peek() {
                Some('|') => {
                    self.scanner.next_char();
                }
                Some(')') => {
                    self.scanner.next_char();
                    return Ok(tokens);
                }
                _ => {
                    let found = self.describe_next();
                    return Err(self.error(DtdErrorKind::Unexpected {
                        expected: "`|` or `)`".to_owned(),
                        found,
                    }));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::DtdParser;
    use crate::error::DtdErrorKind;

    #[test]
    fn comments_and_processing_instructions_are_skipped() -> Result<(), crate::DtdError> {
        let dtd = DtdParser::new()
            .parse_internal_subset("<!-- a comment --><?target data?><!ELEMENT a EMPTY>")?;
        assert!(dtd.element("a").is_some());
        Ok(())
    }

    #[test]
    fn double_hyphen_inside_a_comment_is_rejected() {
        let error = DtdParser::new()
            .parse_internal_subset("<!-- a -- b -->")
            .expect_err("`--` is not allowed inside a comment");
        assert!(matches!(error.kind(), DtdErrorKind::Unexpected { .. }));
    }

    #[test]
    fn conditional_sections_are_rejected_in_the_internal_subset() {
        let error = DtdParser::new()
            .parse_internal_subset("<![INCLUDE[<!ELEMENT a EMPTY>]]>")
            .expect_err("conditional sections are external-subset only");
        assert!(matches!(
            error.kind(),
            DtdErrorKind::ConditionalSectionInInternalSubset
        ));
    }

    #[test]
    fn conditional_sections_work_in_the_external_subset() -> Result<(), crate::DtdError> {
        let dtd = DtdParser::new().parse_external_subset(
            "<![INCLUDE[<!ELEMENT kept EMPTY>]]><![IGNORE[<!ELEMENT dropped EMPTY>]]>",
        )?;
        assert!(dtd.element("kept").is_some());
        assert!(dtd.element("dropped").is_none());
        Ok(())
    }

    #[test]
    fn nested_ignore_sections_are_skipped_as_a_whole() -> Result<(), crate::DtdError> {
        let dtd = DtdParser::new().parse_external_subset(
            "<![IGNORE[ <![INCLUDE[<!ELEMENT a EMPTY>]]> ]]><!ELEMENT b EMPTY>",
        )?;
        assert!(dtd.element("a").is_none());
        assert!(dtd.element("b").is_some());
        Ok(())
    }

    #[test]
    fn parameter_entities_may_not_occur_inside_internal_subset_declarations() {
        let error = DtdParser::new()
            .parse_internal_subset(r#"<!ENTITY % c "EMPTY"><!ELEMENT a %c;>"#)
            .expect_err("WFC: PEs in Internal Subset");
        assert!(matches!(
            error.kind(),
            DtdErrorKind::ParameterEntityInInternalSubset(_)
        ));
    }

    #[test]
    fn parameter_entities_inside_declarations_work_in_the_external_subset()
    -> Result<(), crate::DtdError> {
        let dtd =
            DtdParser::new().parse_external_subset(r#"<!ENTITY % c "EMPTY"><!ELEMENT a %c;>"#)?;
        assert_eq!(
            dtd.element("a")
                .map(|declaration| declaration.content().to_string()),
            Some("EMPTY".to_owned())
        );
        Ok(())
    }

    #[test]
    fn a_quote_from_an_included_parameter_entity_does_not_close_the_literal()
    -> Result<(), crate::DtdError> {
        let dtd =
            DtdParser::new().parse_internal_subset(r#"<!ENTITY % q "&#34;"><!ENTITY e "%q;x">"#)?;
        assert_eq!(
            dtd.general_entity("e").and_then(|e| e.internal_text()),
            Some("\"x")
        );
        Ok(())
    }

    #[test]
    fn general_entity_references_are_bypassed_inside_entity_values() -> Result<(), crate::DtdError>
    {
        let dtd = DtdParser::new().parse_internal_subset(r#"<!ENTITY a "&b;"><!ENTITY b "B">"#)?;
        assert_eq!(
            dtd.general_entity("a").and_then(|e| e.internal_text()),
            Some("&b;")
        );
        Ok(())
    }

    #[test]
    fn improper_parameter_entity_nesting_is_a_validity_error() -> Result<(), crate::DtdError> {
        // VC: Proper Declaration/PE Nesting — not a fatal error, so the element
        // type is still declared and the violation is recorded on the DTD.
        let dtd = DtdParser::new()
            .parse_external_subset(r#"<!ENTITY % tail "EMPTY>"><!ELEMENT a %tail;"#)?;
        assert!(dtd.element("a").is_some());
        assert!(dtd.declaration_errors().iter().any(|error| matches!(
            error.kind(),
            crate::ValidityErrorKind::ImproperDeclarationNesting { .. }
        )));
        Ok(())
    }
}
