//! Grammar checker for the document type declaration.
//!
//! This module implements the productions of [section 2.8] and [section 4] of
//! XML 1.0 that a *non-validating* processor must still enforce: the syntax of
//! the document type declaration and of every markup declaration inside its
//! internal subset. It also collects the general entity declarations, which the
//! [`Reader`](crate::reader::Reader) needs to enforce the
//! [Entity Declared](https://www.w3.org/TR/xml/#wf-entdeclared),
//! [Parsed Entity](https://www.w3.org/TR/xml/#wf-parsedent) and
//! [No Recursion](https://www.w3.org/TR/xml/#norecursion) constraints.
//!
//! Nothing here evaluates *validity* constraints (content models are checked for
//! syntax, never applied to the document).
//!
//! [section 2.8]: https://www.w3.org/TR/xml/#sec-prolog-dtd
//! [section 4]: https://www.w3.org/TR/xml/#sec-physical-struct

use std::collections::HashMap;
use std::sync::Arc;

use crate::XmlVersion;
use crate::errors::WellFormedError;
use crate::wellformed::PREDEFINED_ENTITIES;
use crate::wellformed::chars::{
    is_char, is_literal_char, is_name, is_name_char, is_name_start_char, is_ncname, is_nmtoken,
    is_pubid_char, is_space,
};
use crate::wellformed::limits::{Budget, MAX_ENTITY_DEPTH};

/// A general entity declared by the internal subset of a document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Entity {
    /// An internal entity together with its replacement text, in which character
    /// references have already been expanded (as required by section 4.4.2) while
    /// general entity references have been left alone.
    Internal(String),
    /// An external parsed entity; its replacement text is not available because
    /// this processor never retrieves external entities.
    External,
    /// An unparsed entity, declared with an `NDATA` clause.
    Unparsed,
}

/// Everything a non-validating processor needs to remember about a DTD.
#[derive(Clone, Debug, Default)]
pub(crate) struct Dtd {
    /// General entities declared by the internal subset, by name.
    pub entities: HashMap<String, Entity>,
    /// Whether declarations that this processor cannot see may exist, namely
    /// because the document has an external subset or because the internal
    /// subset contains parameter entity references. When set, the
    /// [Entity Declared] constraint is suspended, as section 4.1 permits.
    ///
    /// Section 4.1 permits that suspension only "in a document without any DTD,
    /// a document with only an internal DTD subset which contains no parameter
    /// entity references, or a document with `standalone='yes'`", so a document
    /// that declares `standalone="yes"` never sets this flag.
    ///
    /// [Entity Declared]: https://www.w3.org/TR/xml/#wf-entdeclared
    pub incomplete: bool,
}

/// Builds an [`InvalidDoctype`](WellFormedError::InvalidDoctype) error.
fn err(message: impl Into<String>) -> WellFormedError {
    WellFormedError::InvalidDoctype(message.into())
}

/// Result alias local to the DTD grammar checker.
type Checked<T> = Result<T, WellFormedError>;

/// Validates the body of a `<!DOCTYPE ...>` declaration, that is the text
/// between `<!DOCTYPE` (and the white space following it) and the closing `>`.
///
/// When `namespace_aware` is set, the names that [Namespaces in XML] requires to
/// be `NCName`s — entity and notation names — are checked as such.
///
/// [Namespaces in XML]: https://www.w3.org/TR/xml-names/
pub(crate) fn validate_doctype(
    body: &str,
    version: XmlVersion,
    namespace_aware: bool,
    standalone: bool,
) -> Checked<Dtd> {
    let mut scanner = Scanner::new(body, version, namespace_aware, standalone);
    let mut dtd = Dtd::default();
    scanner.doctype(&mut dtd)?;
    Ok(dtd)
}

/// A cursor over the text of a document type declaration.
struct Scanner<'a> {
    /// The text being scanned.
    input: &'a str,
    /// Byte offset of the cursor inside [`Self::input`]; always on a char boundary.
    pos: usize,
    /// XML version of the document, which selects the `Char` production.
    version: XmlVersion,
    /// Whether the extra `NCName` constraints of Namespaces in XML apply.
    namespace_aware: bool,
    /// Whether the document declares `standalone="yes"`, in which case the
    /// [Entity Declared] constraint stays in force even with an external subset.
    ///
    /// [Entity Declared]: https://www.w3.org/TR/xml/#wf-entdeclared
    standalone: bool,
    /// Budget bounding the expansion of entities referenced from attribute
    /// defaults declared in this DTD.
    budget: Arc<Budget>,
}

impl<'a> Scanner<'a> {
    /// Creates a scanner positioned at the beginning of `input`.
    fn new(input: &'a str, version: XmlVersion, namespace_aware: bool, standalone: bool) -> Self {
        Self {
            input,
            pos: 0,
            version,
            namespace_aware,
            standalone,
            budget: Budget::new(),
        }
    }

    /// Records that declarations this processor cannot see may exist.
    ///
    /// In a `standalone="yes"` document there are none by definition, so the
    /// [Entity Declared] constraint is not suspended there.
    ///
    /// [Entity Declared]: https://www.w3.org/TR/xml/#wf-entdeclared
    const fn mark_incomplete(&self, dtd: &mut Dtd) {
        if !self.standalone {
            dtd.incomplete = true;
        }
    }

    /// Consumes a `Name` that must additionally be an `NCName` when namespace
    /// processing is active.
    fn ncname(&mut self, what: &str) -> Checked<&'a str> {
        let name = self.name(what)?;
        if self.namespace_aware && !is_ncname(name) {
            return Err(WellFormedError::InvalidQName(name.to_owned()));
        }
        Ok(name)
    }

    /// The not-yet-consumed part of the input.
    #[inline]
    fn rest(&self) -> &'a str {
        &self.input[self.pos..]
    }

    /// The next character, without consuming it.
    #[inline]
    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    /// Consumes and returns the next character.
    #[inline]
    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    /// Consumes `ch` if it is next.
    #[inline]
    fn eat(&mut self, ch: char) -> bool {
        if self.peek() == Some(ch) {
            self.pos += ch.len_utf8();
            true
        } else {
            false
        }
    }

    /// Consumes `text` if the cursor is on it.
    #[inline]
    fn eat_str(&mut self, text: &str) -> bool {
        if self.rest().starts_with(text) {
            self.pos += text.len();
            true
        } else {
            false
        }
    }

    /// Consumes zero or more `S` characters, reporting whether any were found.
    fn skip_space(&mut self) -> bool {
        let start = self.pos;
        while matches!(self.peek(), Some(ch) if is_space(ch)) {
            self.pos += 1;
        }
        self.pos != start
    }

    /// Consumes one or more `S` characters or fails.
    fn require_space(&mut self, what: &str) -> Checked<()> {
        if self.skip_space() {
            Ok(())
        } else {
            Err(err(format!("white space is required {}", what)))
        }
    }

    /// Consumes a `Name` or fails.
    fn name(&mut self, what: &str) -> Checked<&'a str> {
        let start = self.pos;
        match self.peek() {
            Some(ch) if is_name_start_char(ch) => self.pos += ch.len_utf8(),
            _ => return Err(err(format!("a name is expected {}", what))),
        }
        while matches!(self.peek(), Some(ch) if is_name_char(ch)) {
            self.pos += self.peek().map_or(0, char::len_utf8);
        }
        Ok(&self.input[start..self.pos])
    }

    /// Consumes an `Nmtoken` or fails.
    fn nmtoken(&mut self, what: &str) -> Checked<&'a str> {
        let start = self.pos;
        while matches!(self.peek(), Some(ch) if is_name_char(ch)) {
            self.pos += self.peek().map_or(0, char::len_utf8);
        }
        if self.pos == start {
            Err(err(format!("a name token is expected {}", what)))
        } else {
            Ok(&self.input[start..self.pos])
        }
    }

    /// `doctypedecl ::= '<!DOCTYPE' S Name (S ExternalID)? S? ('[' intSubset ']' S?)? '>'`
    ///
    /// The delimiters are stripped by the reader, so only the middle is scanned.
    fn doctype(&mut self, dtd: &mut Dtd) -> Checked<()> {
        self.check_characters()?;
        self.name("as the document type name")?;
        if self.skip_space() && matches!(self.peek(), Some('S' | 'P')) {
            self.external_id()?;
            self.mark_incomplete(dtd);
            self.skip_space();
        }
        if self.eat('[') {
            self.internal_subset(dtd)?;
            if !self.eat(']') {
                return Err(err("`]` is expected at the end of the internal subset"));
            }
            self.skip_space();
        }
        if self.pos != self.input.len() {
            return Err(err(format!(
                "unexpected `{}` after the document type declaration",
                self.snippet()
            )));
        }
        Ok(())
    }

    /// Rejects characters that may not appear literally anywhere in the DTD:
    /// those outside the `Char` production and, in XML 1.1, those of the
    /// [`RestrictedChar`] production.
    ///
    /// [`RestrictedChar`]: https://www.w3.org/TR/xml11/#NT-RestrictedChar
    fn check_characters(&self) -> Checked<()> {
        match self
            .input
            .chars()
            .find(|ch| !is_literal_char(*ch, self.version))
        {
            Some(ch) => Err(WellFormedError::InvalidCharacter(ch)),
            None => Ok(()),
        }
    }

    /// A short excerpt of the remaining input, for error messages.
    fn snippet(&self) -> String {
        self.rest().chars().take(16).collect()
    }

    /// `ExternalID ::= 'SYSTEM' S SystemLiteral | 'PUBLIC' S PubidLiteral S SystemLiteral`
    fn external_id(&mut self) -> Checked<()> {
        if self.eat_str("SYSTEM") {
            self.require_space("after `SYSTEM`")?;
            self.system_literal()?;
            Ok(())
        } else if self.eat_str("PUBLIC") {
            self.require_space("after `PUBLIC`")?;
            self.pubid_literal()?;
            self.require_space("between the public and system identifiers")?;
            self.system_literal()?;
            Ok(())
        } else {
            Err(err(format!(
                "`SYSTEM` or `PUBLIC` is expected, found `{}`",
                self.snippet()
            )))
        }
    }

    /// `ExternalID | PublicID`, as used by a notation declaration.
    fn external_or_public_id(&mut self) -> Checked<()> {
        if self.eat_str("SYSTEM") {
            self.require_space("after `SYSTEM`")?;
            self.system_literal()?;
            Ok(())
        } else if self.eat_str("PUBLIC") {
            self.require_space("after `PUBLIC`")?;
            self.pubid_literal()?;
            // `PublicID` has no system literal, so it is optional here.
            let checkpoint = self.pos;
            if self.skip_space() && matches!(self.peek(), Some('"' | '\'')) {
                self.system_literal()?;
            } else {
                self.pos = checkpoint;
            }
            Ok(())
        } else {
            Err(err(format!(
                "`SYSTEM` or `PUBLIC` is expected, found `{}`",
                self.snippet()
            )))
        }
    }

    /// `SystemLiteral ::= ('"' [^"]* '"') | ("'" [^']* "'")`
    fn system_literal(&mut self) -> Checked<&'a str> {
        let quote = match self.peek() {
            Some(quote @ ('"' | '\'')) => {
                self.pos += 1;
                quote
            }
            _ => return Err(err("a quoted system identifier is expected")),
        };
        let start = self.pos;
        loop {
            match self.bump() {
                Some(ch) if ch == quote => return Ok(&self.input[start..self.pos - 1]),
                Some(_) => {}
                None => return Err(err("the system identifier is not closed")),
            }
        }
    }

    /// `PubidLiteral ::= '"' PubidChar* '"' | "'" (PubidChar - "'")* "'"`
    fn pubid_literal(&mut self) -> Checked<&'a str> {
        let quote = match self.peek() {
            Some(quote @ ('"' | '\'')) => {
                self.pos += 1;
                quote
            }
            _ => return Err(err("a quoted public identifier is expected")),
        };
        let start = self.pos;
        loop {
            match self.bump() {
                Some(ch) if ch == quote => return Ok(&self.input[start..self.pos - 1]),
                Some(ch) if is_pubid_char(ch) => {}
                Some(_) => {
                    let literal = self.input[start..]
                        .split(quote)
                        .next()
                        .unwrap_or_default()
                        .to_owned();
                    return Err(WellFormedError::InvalidPublicId(literal));
                }
                None => return Err(err("the public identifier is not closed")),
            }
        }
    }

    /// `intSubset ::= (markupdecl | DeclSep)*`
    fn internal_subset(&mut self, dtd: &mut Dtd) -> Checked<()> {
        loop {
            self.skip_space();
            match self.peek() {
                None | Some(']') => return Ok(()),
                // `DeclSep ::= PEReference | S`
                Some('%') => {
                    self.pe_reference()?;
                    self.mark_incomplete(dtd);
                }
                Some('<') => self.markup_decl(dtd)?,
                Some(_) => {
                    return Err(err(format!(
                        "a markup declaration is expected, found `{}`",
                        self.snippet()
                    )));
                }
            }
        }
    }

    /// `PEReference ::= '%' Name ';'`
    fn pe_reference(&mut self) -> Checked<()> {
        debug_assert_eq!(self.peek(), Some('%'));
        self.pos += 1;
        self.name("in a parameter entity reference")?;
        if self.eat(';') {
            Ok(())
        } else {
            Err(err(
                "`;` is expected at the end of a parameter entity reference",
            ))
        }
    }

    /// `markupdecl ::= elementdecl | AttlistDecl | EntityDecl | NotationDecl | PI | Comment`
    fn markup_decl(&mut self, dtd: &mut Dtd) -> Checked<()> {
        if self.eat_str("<!--") {
            return self.comment();
        }
        if self.eat_str("<?") {
            return self.processing_instruction();
        }
        if self.eat_str("<!ELEMENT") {
            return self.element_decl();
        }
        if self.eat_str("<!ATTLIST") {
            return self.attlist_decl(dtd);
        }
        if self.eat_str("<!ENTITY") {
            return self.entity_decl(dtd);
        }
        if self.eat_str("<!NOTATION") {
            return self.notation_decl();
        }
        if self.rest().starts_with("<![") {
            return Err(err(
                "conditional sections are only allowed in the external subset",
            ));
        }
        Err(err(format!(
            "unknown markup declaration `{}`",
            self.snippet()
        )))
    }

    /// `Comment ::= '<!--' ((Char - '-') | ('-' (Char - '-')))* '-->'`
    fn comment(&mut self) -> Checked<()> {
        let start = self.pos;
        let Some(end) = self.rest().find("-->") else {
            return Err(err("a comment inside the DTD is not closed"));
        };
        let body = &self.input[start..start + end];
        // The production forbids both an embedded `--` and a trailing `-`
        // (`<!----->` has the body `-`, which matches neither alternative).
        if body.contains("--") {
            return Err(err("`--` is not allowed inside a comment"));
        }
        if body.ends_with('-') {
            return Err(err("a comment cannot end with `-`"));
        }
        self.pos = start + end + 3;
        Ok(())
    }

    /// `PI ::= '<?' PITarget (S (Char* - (Char* '?>' Char*)))? '?>'`
    fn processing_instruction(&mut self) -> Checked<()> {
        let target = self.name("as a processing instruction target")?;
        if target.eq_ignore_ascii_case("xml") {
            return Err(WellFormedError::ReservedPiTarget(target.to_owned()));
        }
        if !self.rest().starts_with("?>") {
            self.require_space("after a processing instruction target")?;
        }
        match self.rest().find("?>") {
            Some(end) => {
                self.pos += end + 2;
                Ok(())
            }
            None => Err(err("a processing instruction inside the DTD is not closed")),
        }
    }

    /// `elementdecl ::= '<!ELEMENT' S Name S contentspec S? '>'`
    fn element_decl(&mut self) -> Checked<()> {
        self.require_space("after `<!ELEMENT`")?;
        self.name("as an element type name")?;
        self.require_space("after an element type name")?;
        self.content_spec()?;
        self.skip_space();
        self.close_decl("<!ELEMENT")
    }

    /// `contentspec ::= 'EMPTY' | 'ANY' | Mixed | children`
    fn content_spec(&mut self) -> Checked<()> {
        if self.eat_str("EMPTY") || self.eat_str("ANY") {
            return Ok(());
        }
        if self.peek() != Some('(') {
            return Err(err(format!(
                "`EMPTY`, `ANY` or a content model is expected, found `{}`",
                self.snippet()
            )));
        }
        // Distinguish `Mixed` from `children` by looking for `#PCDATA`.
        let checkpoint = self.pos;
        self.pos += 1;
        self.skip_space();
        if self.rest().starts_with("#PCDATA") {
            self.pos = checkpoint;
            self.mixed()
        } else {
            self.pos = checkpoint;
            self.children()
        }
    }

    /// ```text
    /// Mixed ::= '(' S? '#PCDATA' (S? '|' S? Name)* S? ')*'
    ///         | '(' S? '#PCDATA' S? ')'
    /// ```
    fn mixed(&mut self) -> Checked<()> {
        debug_assert_eq!(self.peek(), Some('('));
        self.pos += 1;
        self.skip_space();
        if !self.eat_str("#PCDATA") {
            return Err(err(
                "`#PCDATA` is expected at the start of a mixed content model",
            ));
        }
        let mut has_names = false;
        loop {
            self.skip_space();
            if self.eat('|') {
                has_names = true;
                self.skip_space();
                self.name("in a mixed content model")?;
            } else {
                break;
            }
        }
        if !self.eat(')') {
            return Err(err(format!(
                "`)` is expected to close a mixed content model, found `{}`",
                self.snippet()
            )));
        }
        if has_names {
            if !self.eat('*') {
                return Err(err("`*` is required after `(#PCDATA | ...)`"));
            }
        } else {
            // `(#PCDATA)` may optionally be followed by `*`, but by nothing else.
            self.eat('*');
        }
        Ok(())
    }

    /// `children ::= (choice | seq) ('?' | '*' | '+')?`
    fn children(&mut self) -> Checked<()> {
        self.group()?;
        self.occurrence();
        Ok(())
    }

    /// Parses a `choice` or a `seq`, which differ only by their separator.
    ///
    /// ```text
    /// choice ::= '(' S? cp ( S? '|' S? cp )+ S? ')'
    /// seq    ::= '(' S? cp ( S? ',' S? cp )* S? ')'
    /// ```
    fn group(&mut self) -> Checked<()> {
        if !self.eat('(') {
            return Err(err(format!(
                "`(` is expected to start a content model, found `{}`",
                self.snippet()
            )));
        }
        self.skip_space();
        self.content_particle()?;
        let mut separator: Option<char> = None;
        loop {
            self.skip_space();
            match self.peek() {
                Some(sep @ ('|' | ',')) => {
                    match separator {
                        None => separator = Some(sep),
                        Some(expected) if expected == sep => {}
                        Some(expected) => {
                            return Err(err(format!(
                                "a content model cannot mix the `{}` and `{}` connectors",
                                expected, sep
                            )));
                        }
                    }
                    self.pos += 1;
                    self.skip_space();
                    self.content_particle()?;
                }
                Some(')') => {
                    self.pos += 1;
                    return Ok(());
                }
                _ => {
                    return Err(err(format!(
                        "`|`, `,` or `)` is expected in a content model, found `{}`",
                        self.snippet()
                    )));
                }
            }
        }
    }

    /// `cp ::= (Name | choice | seq) ('?' | '*' | '+')?`
    fn content_particle(&mut self) -> Checked<()> {
        if self.peek() == Some('(') {
            self.group()?;
        } else {
            self.name("in a content model")?;
        }
        self.occurrence();
        Ok(())
    }

    /// Consumes an optional `?`, `*` or `+` occurrence indicator.
    fn occurrence(&mut self) {
        let _ = self.eat('?') || self.eat('*') || self.eat('+');
    }

    /// `AttlistDecl ::= '<!ATTLIST' S Name AttDef* S? '>'`
    fn attlist_decl(&mut self, dtd: &Dtd) -> Checked<()> {
        self.require_space("after `<!ATTLIST`")?;
        self.name("as an element type name")?;
        loop {
            let checkpoint = self.pos;
            if !self.skip_space() {
                break;
            }
            if matches!(self.peek(), None | Some('>')) {
                self.pos = checkpoint;
                break;
            }
            self.att_def(dtd)?;
        }
        self.skip_space();
        self.close_decl("<!ATTLIST")
    }

    /// `AttDef ::= S Name S AttType S DefaultDecl`
    fn att_def(&mut self, dtd: &Dtd) -> Checked<()> {
        self.name("as an attribute name")?;
        self.require_space("after an attribute name")?;
        self.att_type()?;
        self.require_space("before an attribute default declaration")?;
        self.default_decl(dtd)
    }

    /// ```text
    /// AttType        ::= StringType | TokenizedType | EnumeratedType
    /// StringType     ::= 'CDATA'
    /// TokenizedType  ::= 'ID' | 'IDREF' | 'IDREFS' | 'ENTITY' | 'ENTITIES'
    ///                  | 'NMTOKEN' | 'NMTOKENS'
    /// EnumeratedType ::= NotationType | Enumeration
    /// ```
    fn att_type(&mut self) -> Checked<()> {
        // Longest match first, so that `IDREFS` is not read as `IDREF`.
        for keyword in [
            "CDATA", "IDREFS", "IDREF", "ID", "ENTITIES", "ENTITY", "NMTOKENS", "NMTOKEN",
        ] {
            if self.rest().starts_with(keyword) {
                let after = self.input[self.pos + keyword.len()..].chars().next();
                if matches!(after, Some(ch) if is_name_char(ch)) {
                    continue;
                }
                self.pos += keyword.len();
                return Ok(());
            }
        }
        if self.eat_str("NOTATION") {
            self.require_space("after `NOTATION`")?;
            return self.enumeration(true);
        }
        if self.peek() == Some('(') {
            return self.enumeration(false);
        }
        Err(err(format!(
            "an attribute type is expected, found `{}`",
            self.snippet()
        )))
    }

    /// ```text
    /// NotationType ::= 'NOTATION' S '(' S? Name (S? '|' S? Name)* S? ')'
    /// Enumeration  ::= '(' S? Nmtoken (S? '|' S? Nmtoken)* S? ')'
    /// ```
    fn enumeration(&mut self, names_only: bool) -> Checked<()> {
        if !self.eat('(') {
            return Err(err("`(` is expected to start an enumerated attribute type"));
        }
        loop {
            self.skip_space();
            if names_only {
                self.name("in a notation attribute type")?;
            } else {
                let token = self.nmtoken("in an enumerated attribute type")?;
                if !is_nmtoken(token) {
                    return Err(WellFormedError::InvalidName(token.to_owned()));
                }
            }
            self.skip_space();
            if self.eat('|') {
                continue;
            }
            if self.eat(')') {
                return Ok(());
            }
            return Err(err(format!(
                "`|` or `)` is expected in an enumerated attribute type, found `{}`",
                self.snippet()
            )));
        }
    }

    /// `DefaultDecl ::= '#REQUIRED' | '#IMPLIED' | (('#FIXED' S)? AttValue)`
    fn default_decl(&mut self, dtd: &Dtd) -> Checked<()> {
        if self.eat_str("#REQUIRED") || self.eat_str("#IMPLIED") {
            return Ok(());
        }
        if self.eat_str("#FIXED") {
            self.require_space("after `#FIXED`")?;
        }
        let value = self.att_value()?;
        // The `Entity Declared` constraint requires every entity referenced from
        // an attribute default to have been declared *before* the reference.
        let mut stack = Vec::new();
        check_attribute_value(value, dtd, self.version, &mut stack, &self.budget)
    }

    /// `AttValue ::= '"' ([^<&"] | Reference)* '"' | "'" ([^<&'] | Reference)* "'"`
    fn att_value(&mut self) -> Checked<&'a str> {
        let quote = match self.peek() {
            Some(quote @ ('"' | '\'')) => {
                self.pos += 1;
                quote
            }
            _ => {
                return Err(err(format!(
                    "a quoted attribute value is expected, found `{}`",
                    self.snippet()
                )));
            }
        };
        let start = self.pos;
        loop {
            match self.bump() {
                Some(ch) if ch == quote => return Ok(&self.input[start..self.pos - 1]),
                Some('<') => {
                    return Err(err("`<` is not allowed in an attribute value"));
                }
                Some('&') => {
                    self.reference()?;
                }
                Some(_) => {}
                None => return Err(err("an attribute value is not closed")),
            }
        }
    }

    /// `EntityDecl ::= GEDecl | PEDecl`
    fn entity_decl(&mut self, dtd: &mut Dtd) -> Checked<()> {
        self.require_space("after `<!ENTITY`")?;
        let parameter = self.eat('%');
        if parameter {
            self.require_space("after `%` in a parameter entity declaration")?;
            self.mark_incomplete(dtd);
        }
        let name = self.ncname("as an entity name")?.to_owned();
        self.require_space("after an entity name")?;
        match self.peek() {
            Some('"' | '\'') => {
                let replacement = self.entity_value()?;
                if !parameter {
                    dtd.entities
                        .entry(name)
                        .or_insert(Entity::Internal(replacement));
                }
            }
            _ => {
                self.external_id()?;
                let mut unparsed = false;
                let checkpoint = self.pos;
                if self.skip_space() {
                    if self.eat_str("NDATA") {
                        if parameter {
                            return Err(err(
                                "`NDATA` is not allowed in a parameter entity declaration",
                            ));
                        }
                        self.require_space("after `NDATA`")?;
                        self.name("as a notation name")?;
                        unparsed = true;
                    } else {
                        self.pos = checkpoint;
                    }
                }
                if !parameter {
                    let entity = if unparsed {
                        Entity::Unparsed
                    } else {
                        Entity::External
                    };
                    dtd.entities.entry(name).or_insert(entity);
                }
            }
        }
        self.skip_space();
        self.close_decl("<!ENTITY")
    }

    /// ```text
    /// EntityValue ::= '"' ([^%&"] | PEReference | Reference)* '"'
    ///               | "'" ([^%&'] | PEReference | Reference)* "'"
    /// ```
    ///
    /// Returns the replacement text with character references expanded.
    ///
    /// Parameter entity references are rejected: this scanner only ever sees the
    /// internal subset, where the [PEs in Internal Subset] constraint forbids a
    /// parameter entity reference from occurring inside a markup declaration.
    ///
    /// [PEs in Internal Subset]: https://www.w3.org/TR/xml/#wfc-PEinInternalSubset
    fn entity_value(&mut self) -> Checked<String> {
        let quote = match self.peek() {
            Some(quote @ ('"' | '\'')) => {
                self.pos += 1;
                quote
            }
            _ => return Err(err("a quoted entity value is expected")),
        };
        let mut replacement = String::new();
        loop {
            match self.bump() {
                Some(ch) if ch == quote => return Ok(replacement),
                Some('%') => {
                    return Err(err(
                        "a parameter entity reference cannot occur inside a markup declaration of the internal subset",
                    ));
                }
                Some('&') => {
                    let start = self.pos - 1;
                    match self.reference()? {
                        // Character references are expanded immediately.
                        Some(ch) => replacement.push(ch),
                        // General entity references are bypassed.
                        None => replacement.push_str(&self.input[start..self.pos]),
                    }
                }
                Some(ch) => replacement.push(ch),
                None => return Err(err("an entity value is not closed")),
            }
        }
    }

    /// `Reference ::= EntityRef | CharRef`, with the leading `&` already consumed.
    ///
    /// Returns the denoted character for a `CharRef`, and `None` for an
    /// `EntityRef`.
    fn reference(&mut self) -> Checked<Option<char>> {
        let start = self.pos;
        if self.eat('#') {
            let radix = if self.eat('x') { 16 } else { 10 };
            let digits_start = self.pos;
            while matches!(self.peek(), Some(ch) if ch.is_digit(radix)) {
                self.pos += 1;
            }
            let digits = &self.input[digits_start..self.pos];
            if digits.is_empty() || !self.eat(';') {
                return Err(WellFormedError::InvalidReference(format!(
                    "&{}",
                    self.input[start..].chars().take(12).collect::<String>()
                )));
            }
            let code = u32::from_str_radix(digits, radix).ok();
            return match code.and_then(char::from_u32) {
                Some(ch) if is_char(ch, self.version) => Ok(Some(ch)),
                _ => Err(WellFormedError::InvalidReference(
                    self.input[start - 1..self.pos].to_owned(),
                )),
            };
        }
        self.name("in an entity reference")?;
        if self.eat(';') {
            Ok(None)
        } else {
            Err(WellFormedError::InvalidReference(
                self.input[start - 1..self.pos].to_owned(),
            ))
        }
    }

    /// `NotationDecl ::= '<!NOTATION' S Name S (ExternalID | PublicID) S? '>'`
    fn notation_decl(&mut self) -> Checked<()> {
        self.require_space("after `<!NOTATION`")?;
        self.ncname("as a notation name")?;
        self.require_space("after a notation name")?;
        self.external_or_public_id()?;
        self.skip_space();
        self.close_decl("<!NOTATION")
    }

    /// Consumes the `>` that terminates a markup declaration.
    fn close_decl(&mut self, what: &str) -> Checked<()> {
        if self.eat('>') {
            Ok(())
        } else {
            Err(err(format!(
                "`>` is expected to close a `{}` declaration, found `{}`",
                what,
                self.snippet()
            )))
        }
    }
}

/// Validates every `&...;` reference of an attribute value, expanding internal
/// general entities so that the constraints apply to their replacement text too.
///
/// `stack` carries the names of the entities currently being expanded and is used
/// to detect the recursion that the [No Recursion] constraint forbids.
///
/// [No Recursion]: https://www.w3.org/TR/xml/#norecursion
pub(crate) fn check_attribute_value(
    value: &str,
    dtd: &Dtd,
    version: XmlVersion,
    stack: &mut Vec<String>,
    budget: &Budget,
) -> Checked<()> {
    let mut rest = value;
    while let Some(at) = rest.find('&') {
        rest = &rest[at + 1..];
        let Some(end) = rest.find(';') else {
            return Err(WellFormedError::InvalidReference(format!(
                "&{}",
                rest.chars().take(12).collect::<String>()
            )));
        };
        let (name, tail) = rest.split_at(end);
        check_attribute_reference(name, dtd, version, stack, budget)?;
        rest = &tail[1..];
    }
    Ok(())
}

/// Validates one reference of an attribute value, given the text between `&`
/// and `;`.
fn check_attribute_reference(
    name: &str,
    dtd: &Dtd,
    version: XmlVersion,
    stack: &mut Vec<String>,
    budget: &Budget,
) -> Checked<()> {
    if name.starts_with('#') {
        return check_character_reference(name, version);
    }
    if !is_name(name) {
        return Err(WellFormedError::InvalidName(name.to_owned()));
    }
    if PREDEFINED_ENTITIES.contains(&name) {
        return Ok(());
    }
    match dtd.entities.get(name) {
        Some(Entity::Unparsed) => Err(WellFormedError::UnparsedEntityReference(name.to_owned())),
        Some(Entity::External) => Err(WellFormedError::ExternalEntityInAttributeValue(
            name.to_owned(),
        )),
        Some(Entity::Internal(replacement)) => {
            if stack.iter().any(|open| open == name) || stack.len() >= MAX_ENTITY_DEPTH {
                return Err(WellFormedError::RecursiveEntity(name.to_owned()));
            }
            if replacement.contains('<') {
                return Err(WellFormedError::LessThanInAttributeValue(name.to_owned()));
            }
            budget.charge(name, replacement.len())?;
            let replacement = replacement.clone();
            stack.push(name.to_owned());
            let outcome = check_attribute_value(&replacement, dtd, version, stack, budget);
            stack.pop();
            outcome.map_err(|error| match error {
                // Exhausting the expansion budget is a property of the document
                // as a whole, so it is reported as such instead of being wrapped
                // once per nesting level.
                limit @ WellFormedError::EntityExpansionLimit { .. } => limit,
                other => WellFormedError::IllFormedEntity {
                    entity: name.to_owned(),
                    reason: other.to_string(),
                },
            })
        }
        None if dtd.incomplete => Ok(()),
        None => Err(WellFormedError::UndeclaredEntity(name.to_owned())),
    }
}

/// Validates a `&#...;` character reference, given the text between `&` and `;`.
pub(crate) fn check_character_reference(name: &str, version: XmlVersion) -> Checked<()> {
    let digits = name.strip_prefix('#').unwrap_or(name);
    let (digits, radix) = match digits.strip_prefix('x') {
        Some(hex) => (hex, 16),
        None => (digits, 10),
    };
    let valid = !digits.is_empty()
        && digits.chars().all(|ch| ch.is_digit(radix))
        && u32::from_str_radix(digits, radix)
            .ok()
            .and_then(char::from_u32)
            .is_some_and(|ch| is_char(ch, version));
    if valid {
        Ok(())
    } else {
        Err(WellFormedError::InvalidReference(format!("&{};", name)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shorthand that validates a doctype body for XML 1.0.
    fn check(body: &str) -> Result<Dtd, WellFormedError> {
        validate_doctype(body, XmlVersion::Implicit1_0, false, false)
    }

    /// Shorthand that validates a doctype body of a `standalone="yes"` document.
    fn check_standalone(body: &str) -> Result<Dtd, WellFormedError> {
        validate_doctype(body, XmlVersion::Implicit1_0, false, true)
    }

    #[test]
    fn minimal() -> Result<(), WellFormedError> {
        let dtd = check("doc")?;
        assert!(dtd.entities.is_empty());
        assert!(!dtd.incomplete);
        Ok(())
    }

    #[test]
    fn external_subset_marks_incomplete() -> Result<(), WellFormedError> {
        assert!(check("doc SYSTEM \"doc.dtd\"")?.incomplete);
        assert!(check("doc PUBLIC \"-//X//DTD//EN\" \"doc.dtd\"")?.incomplete);
        Ok(())
    }

    /// Section 4.1 keeps the Entity Declared constraint in force for a
    /// `standalone="yes"` document, so nothing marks its DTD as incomplete.
    #[test]
    fn standalone_documents_are_never_incomplete() -> Result<(), WellFormedError> {
        assert!(!check_standalone("doc SYSTEM \"doc.dtd\"")?.incomplete);
        assert!(!check_standalone("doc PUBLIC \"-//X//DTD//EN\" \"doc.dtd\"")?.incomplete);
        assert!(!check_standalone("doc [<!ENTITY % p SYSTEM 'p'>%p;]")?.incomplete);
        assert!(check("doc [<!ENTITY % p SYSTEM 'p'>%p;]")?.incomplete);
        Ok(())
    }

    #[test]
    fn collects_entities() -> Result<(), WellFormedError> {
        let dtd = check(
            "doc [<!ENTITY e \"&#60;text\"><!NOTATION n SYSTEM \"n\"><!ENTITY u SYSTEM \"u\" NDATA n>]",
        )?;
        assert_eq!(
            dtd.entities.get("e"),
            Some(&Entity::Internal("<text".into()))
        );
        assert_eq!(dtd.entities.get("u"), Some(&Entity::Unparsed));
        Ok(())
    }

    #[test]
    fn rejects_bad_declarations() {
        for body in [
            "doc [<!DUNNO x>]",
            "doc [<!ELEMENT doc CDATA>]",
            "doc [<!ELEMENT doc (a, (b) | c)?>]",
            "doc [<!ELEMENT doc ((#PCDATA))>]",
            "doc [<!ELEMENT doc (#PCDATA)+>]",
            "doc [<!ELEMENT doc (a *)>]",
            "doc [<!ELEMENT doc (a) *>]",
            "doc [<!ATTLIST doc a1 (foo,bar) #IMPLIED>]",
            "doc [<!ATTLIST doc a1 NAME #IMPLIED>]",
            "doc [<!ATTLIST doc a1(foo|bar) #IMPLIED>]",
            "doc [<!ENTITY foo PUBLIC \"some public id\">]",
            "doc [<!ENTITY foo\"some text\">]",
            "doc [<!ENTITY foo \"&\">]",
            "doc [<!ENTITY #DEFAULT \"default\">]",
            "doc [<![INCLUDE[ ]]>]",
            "doc [<!NOTATION not1 PUBLIC '\"'>]",
            "doc -- a comment -- []",
            "doc [<!ELEMENT doc EMPTY><doc/>]",
            "doc [<!ENTITY % e \"\"><!ENTITY foo \"%e;\">]",
            "doc [<!ELEMENT doc (#PCDATA)><!ATTLIST doc a CDATA \"&e;\"><!ENTITY e 'v'>]",
            "doc [<!ELEMENT doc (#PCDATA)><!ENTITY e SYSTEM 'e'><!ATTLIST doc a CDATA \"&e;\">]",
        ] {
            assert!(check(body).is_err(), "expected an error for `{body}`");
        }
    }

    #[test]
    fn accepts_good_declarations() -> Result<(), WellFormedError> {
        for body in [
            "doc [<!ELEMENT doc EMPTY>]",
            "doc [<!ELEMENT doc ANY>]",
            "doc [<!ELEMENT doc (#PCDATA)>]",
            "doc [<!ELEMENT doc (#PCDATA | a | b)*>]",
            "doc [<!ELEMENT doc (a, b?, (c | d)+)*>]",
            "doc [<!ATTLIST doc a CDATA #IMPLIED b (x | y) 'x' c NOTATION (n) #REQUIRED>]",
            "doc [<!NOTATION n SYSTEM 'n'><!ENTITY e SYSTEM 'e' NDATA n>]",
            "doc [<!-- comment --><?pi data?>]",
            "doc [<!ENTITY % p 'text'>]",
            "doc [<!ELEMENT doc (#PCDATA)><!ENTITY e 'v'><!ATTLIST doc a CDATA \"&e;\">]",
        ] {
            check(body)?;
        }
        Ok(())
    }
}
