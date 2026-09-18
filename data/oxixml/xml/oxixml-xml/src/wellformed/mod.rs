//! Optional, conformance-grade well-formedness checking.
//!
//! By default this crate parses exactly like `quick-xml` 0.41: the lexer is
//! permissive and only reports the handful of problems that `quick-xml` reports.
//! Setting [`Config::check_well_formedness`] activates the checks implemented
//! here, which together turn the [`Reader`] into a conforming **non-validating**
//! XML processor:
//!
//! * the [`Char`] production is enforced on every piece of character data,
//!   comment, processing instruction and attribute value;
//! * the [`Name`] production is enforced on element, attribute, entity and
//!   processing-instruction-target names;
//! * the document structure of [section 2.1] is enforced (a single root element,
//!   the XML declaration only at the very beginning, at most one document type
//!   declaration before the root element, and no character data outside the root
//!   element);
//! * the [`XMLDecl`] and [`doctypedecl`] productions — including the whole
//!   grammar of the internal subset — are enforced;
//! * the [Entity Declared], [Parsed Entity], [No Recursion] and
//!   [No `<` in Attribute Values] constraints are enforced.
//!
//! *Validity* constraints (content models, attribute types, ID uniqueness) are
//! deliberately **not** enforced: they require a validating processor.
//!
//! [`Config::check_well_formedness`]: crate::reader::Config::check_well_formedness
//! [`Reader`]: crate::reader::Reader
//! [`Char`]: https://www.w3.org/TR/xml/#NT-Char
//! [`Name`]: https://www.w3.org/TR/xml/#NT-Name
//! [`XMLDecl`]: https://www.w3.org/TR/xml/#NT-XMLDecl
//! [`doctypedecl`]: https://www.w3.org/TR/xml/#NT-doctypedecl
//! [section 2.1]: https://www.w3.org/TR/xml/#sec-well-formed
//! [Entity Declared]: https://www.w3.org/TR/xml/#wf-entdeclared
//! [Parsed Entity]: https://www.w3.org/TR/xml/#wf-parsedent
//! [No Recursion]: https://www.w3.org/TR/xml/#norecursion
//! [No `<` in Attribute Values]: https://www.w3.org/TR/xml/#CleanAttrVals

pub(crate) mod chars;
pub(crate) mod decl;
pub(crate) mod dtd;
pub(crate) mod limits;

use std::sync::Arc;

use crate::XmlVersion;
use crate::encoding::Decoder;
use crate::errors::{Error, Result, WellFormedError};
use crate::events::Event;
use crate::wellformed::chars::{
    is_literal_char, is_name, is_name_char, is_name_start_char, is_ncname, is_qname, is_space,
};
use crate::wellformed::dtd::{Dtd, Entity};
use crate::wellformed::limits::{Budget, MAX_ENTITY_DEPTH};

/// The five entities that XML predefines and that therefore never need to be
/// declared: <https://www.w3.org/TR/xml/#sec-predefined-ent>.
pub(crate) const PREDEFINED_ENTITIES: [&str; 5] = ["amp", "lt", "gt", "apos", "quot"];

/// Incremental well-formedness checker driven by the [`Reader`](crate::reader::Reader).
///
/// One instance follows one entity: the outer instance follows the document
/// entity, and [`Self::nested`] instances follow the replacement text of general
/// entities.
#[derive(Clone, Debug)]
pub(crate) struct WellFormed {
    /// Declarations gathered from the document type declaration.
    dtd: Arc<Dtd>,
    /// Names of the general entities currently being expanded, innermost last.
    stack: Vec<String>,
    /// Budget shared by every checker of this document, which bounds the total
    /// amount of entity expansion (see [`limits`]).
    budget: Arc<Budget>,
    /// XML version of the document, taken from the XML declaration.
    version: XmlVersion,
    /// Number of open elements.
    depth: usize,
    /// Whether a root element has been seen.
    seen_root: bool,
    /// Whether the document declares `standalone="yes"`.
    standalone: bool,
    /// Whether a document type declaration has been seen.
    seen_doctype: bool,
    /// Whether any event at all has been seen, used to place the XML declaration.
    seen_event: bool,
    /// Whether this checker follows the replacement text of an entity rather
    /// than the document entity. Fragments have no root element and may contain
    /// character data at the top level.
    fragment: bool,
    /// Whether the document is processed by a namespace-aware reader, in which
    /// case the extra name constraints of [Namespaces in XML] apply.
    ///
    /// [Namespaces in XML]: https://www.w3.org/TR/xml-names/
    namespace_aware: bool,
}

impl Default for WellFormed {
    fn default() -> Self {
        Self {
            dtd: Arc::new(Dtd::default()),
            stack: Vec::new(),
            budget: Budget::new(),
            version: XmlVersion::Implicit1_0,
            standalone: false,
            depth: 0,
            seen_root: false,
            seen_doctype: false,
            seen_event: false,
            fragment: false,
            namespace_aware: false,
        }
    }
}

impl WellFormed {
    /// Number of currently open elements.
    #[inline]
    pub(crate) const fn depth(&self) -> usize {
        self.depth
    }

    /// Creates a checker for the replacement text of an entity, sharing this
    /// checker's declarations and expansion stack.
    fn nested(&self) -> Self {
        Self {
            dtd: Arc::clone(&self.dtd),
            stack: self.stack.clone(),
            budget: Arc::clone(&self.budget),
            version: self.version,
            standalone: self.standalone,
            depth: 0,
            seen_root: false,
            seen_doctype: false,
            seen_event: true,
            fragment: true,
            namespace_aware: self.namespace_aware,
        }
    }

    /// Returns the replacement text of an internal general entity declared by the
    /// document's internal subset, if any.
    #[inline]
    pub(crate) fn declared_entity(&self, name: &str) -> Option<&str> {
        match self.dtd.entities.get(name) {
            Some(Entity::Internal(replacement)) => Some(replacement),
            _ => None,
        }
    }

    /// Switches on the extra name constraints of Namespaces in XML.
    #[inline]
    pub(crate) const fn set_namespace_aware(&mut self) {
        self.namespace_aware = true;
    }

    /// Rejects a name that must be an `NCName` while namespace processing is on.
    fn require_ncname(&self, name: &str) -> Result<()> {
        if !self.namespace_aware || is_ncname(name) {
            Ok(())
        } else {
            Err(WellFormedError::InvalidQName(name.to_owned()).into())
        }
    }

    /// Rejects a name that must be a `QName` while namespace processing is on.
    fn require_qname(&self, name: &str) -> Result<()> {
        if !self.namespace_aware || is_qname(name) {
            Ok(())
        } else {
            Err(WellFormedError::InvalidQName(name.to_owned()).into())
        }
    }

    /// Validates one event. Called for every event the reader produces while
    /// [`Config::check_well_formedness`](crate::reader::Config::check_well_formedness)
    /// is enabled.
    pub(crate) fn check(&mut self, event: &Event<'_>, decoder: Decoder) -> Result<()> {
        let was_first = !self.seen_event;
        self.seen_event = true;
        match event {
            Event::Decl(declaration) => {
                if !was_first || self.fragment {
                    return Err(WellFormedError::MisplacedXmlDeclaration.into());
                }
                let text = decoder.decode(declaration)?;
                let declaration = decl::validate(&text)?;
                self.version = declaration.version;
                self.standalone = declaration.standalone;
                Ok(())
            }
            Event::DocType(doctype) => {
                if self.fragment || self.seen_doctype || self.seen_root || self.depth > 0 {
                    return Err(WellFormedError::MisplacedDoctype.into());
                }
                self.seen_doctype = true;
                let text = decoder.decode(doctype)?;
                self.dtd = Arc::new(dtd::validate_doctype(
                    &text,
                    self.version,
                    self.namespace_aware,
                    self.standalone,
                )?);
                Ok(())
            }
            Event::Start(start) => {
                self.enter_element()?;
                self.depth += 1;
                self.check_tag(start, decoder)
            }
            Event::Empty(start) => {
                self.enter_element()?;
                self.check_tag(start, decoder)
            }
            Event::End(end) => {
                self.depth = self.depth.saturating_sub(1);
                let end_name = end.name();
                let name = decoder.decode(end_name.as_ref())?;
                require_name(&name)?;
                self.require_qname(&name)
            }
            Event::Text(text) => {
                let decoded = decoder.decode(text)?;
                self.check_characters(&decoded)?;
                if decoded.contains("]]>") {
                    return Err(WellFormedError::CDataEndInContent.into());
                }
                if !self.fragment && self.depth == 0 && !decoded.chars().all(is_space) {
                    return Err(WellFormedError::ContentOutsideRoot.into());
                }
                Ok(())
            }
            Event::CData(cdata) => {
                if !self.fragment && self.depth == 0 {
                    return Err(WellFormedError::ContentOutsideRoot.into());
                }
                let decoded = decoder.decode(cdata)?;
                self.check_characters(&decoded)
            }
            Event::Comment(comment) => {
                let decoded = decoder.decode(comment)?;
                self.check_characters(&decoded)?;
                if decoded.contains("--") || decoded.ends_with('-') {
                    return Err(Error::IllFormed(
                        crate::errors::IllFormedError::DoubleHyphenInComment,
                    ));
                }
                Ok(())
            }
            Event::PI(instruction) => {
                let target = decoder.decode(instruction.target())?;
                require_name(&target)?;
                self.require_ncname(&target)?;
                if target.eq_ignore_ascii_case("xml") {
                    return Err(WellFormedError::ReservedPiTarget(target.into_owned()).into());
                }
                let content = decoder.decode(instruction.content())?;
                if !content.is_empty() && !content.starts_with(is_space) {
                    return Err(WellFormedError::InvalidName(target.into_owned()).into());
                }
                self.check_characters(&content)
            }
            Event::GeneralRef(reference) => {
                if !self.fragment && self.depth == 0 {
                    return Err(WellFormedError::ContentOutsideRoot.into());
                }
                let name = decoder.decode(reference)?;
                if !name.starts_with('#') {
                    self.require_ncname(&name)?;
                }
                self.check_reference(&name)
            }
            Event::Eof => {
                if !self.fragment && !self.seen_root {
                    return Err(WellFormedError::MissingRootElement.into());
                }
                Ok(())
            }
        }
    }

    /// Records that a new element starts, rejecting a second root element.
    fn enter_element(&mut self) -> Result<()> {
        if self.depth == 0 && !self.fragment {
            if self.seen_root {
                return Err(WellFormedError::MultipleRootElements.into());
            }
            self.seen_root = true;
        }
        Ok(())
    }

    /// Validates a start or empty-element tag against the [`STag`] production.
    ///
    /// ```text
    /// STag      ::= '<' Name (S Attribute)* S? '>'
    /// Attribute ::= Name Eq AttValue
    /// ```
    ///
    /// The reader hands over the text between `<` and `>` (with a trailing `/`
    /// already removed for empty-element tags), so only the middle is scanned.
    ///
    /// [`STag`]: https://www.w3.org/TR/xml/#NT-STag
    fn check_tag(&mut self, start: &crate::events::BytesStart<'_>, decoder: Decoder) -> Result<()> {
        let content = decoder.decode(start)?;
        self.check_characters(&content)?;
        let mut cursor = TagCursor {
            input: &content,
            pos: 0,
        };
        let element = cursor.name()?;
        self.require_qname(element)?;
        let mut seen: Vec<&str> = Vec::new();
        while cursor.skip_space() && !cursor.is_empty() {
            let key = cursor.name()?;
            if seen.contains(&key) {
                return Err(WellFormedError::DuplicateAttribute(key.to_owned()).into());
            }
            seen.push(key);
            self.require_qname(key)?;
            cursor.eq(key)?;
            let value = cursor.att_value(key)?;
            // Namespaces in XML 1.0 does not allow a prefix to be unbound, while
            // section 6.1 of Namespaces in XML 1.1 introduces exactly that:
            // `xmlns:p=""` undeclares the prefix `p`.
            if self.namespace_aware
                && !matches!(self.version, XmlVersion::Explicit1_1)
                && value.is_empty()
                && let Some(prefix) = key.strip_prefix("xmlns:")
            {
                return Err(WellFormedError::EmptyPrefixBinding(prefix.to_owned()).into());
            }
            self.check_value_references(value)?;
        }
        if cursor.is_empty() {
            Ok(())
        } else {
            Err(WellFormedError::MalformedTag(format!(
                "white space is required before the attribute `{}`",
                cursor.snippet()
            ))
            .into())
        }
    }

    /// Rejects characters that may not appear literally in the document:
    /// everything outside the `Char` production of the document version and, for
    /// XML 1.1, everything inside its [`RestrictedChar`] production.
    ///
    /// The same code points remain legal when they are written as a character
    /// reference, which [`dtd::check_character_reference`] validates instead.
    ///
    /// Replacement text of an entity (`self.fragment`) is exempt: character
    /// references inside an entity value have already been expanded when the
    /// declaration was read, so what is checked here is no longer literal text.
    ///
    /// [`RestrictedChar`]: https://www.w3.org/TR/xml11/#NT-RestrictedChar
    fn check_characters(&self, text: &str) -> Result<()> {
        let version = self.version;
        let invalid = if self.fragment {
            text.chars().find(|ch| !chars::is_char(*ch, version))
        } else {
            text.chars().find(|ch| !is_literal_char(*ch, version))
        };
        match invalid {
            Some(ch) => Err(WellFormedError::InvalidCharacter(ch).into()),
            None => Ok(()),
        }
    }

    /// Validates every `&...;` reference inside an attribute value.
    fn check_value_references(&mut self, value: &str) -> Result<()> {
        let dtd = Arc::clone(&self.dtd);
        let budget = Arc::clone(&self.budget);
        dtd::check_attribute_value(value, &dtd, self.version, &mut self.stack, &budget)?;
        Ok(())
    }

    /// Validates a reference found in element content, given the text between
    /// `&` and `;`.
    fn check_reference(&mut self, name: &str) -> Result<()> {
        if name.starts_with('#') {
            dtd::check_character_reference(name, self.version)?;
            return Ok(());
        }
        require_name(name)?;
        if PREDEFINED_ENTITIES.contains(&name) {
            return Ok(());
        }
        match self.dtd.entities.get(name) {
            Some(Entity::Unparsed) => {
                Err(WellFormedError::UnparsedEntityReference(name.to_owned()).into())
            }
            // The replacement text of an external entity is never read, so no
            // further constraint can be checked.
            Some(Entity::External) => Ok(()),
            Some(Entity::Internal(replacement)) => {
                if self.stack.iter().any(|open| open == name)
                    || self.stack.len() >= MAX_ENTITY_DEPTH
                {
                    return Err(WellFormedError::RecursiveEntity(name.to_owned()).into());
                }
                self.budget.charge(name, replacement.len())?;
                let replacement = replacement.clone();
                self.stack.push(name.to_owned());
                let outcome = self.check_replacement_content(&replacement);
                self.stack.pop();
                outcome.map_err(|error| match error {
                    // Exhausting the expansion budget is a property of the
                    // document as a whole, so it is reported as such instead of
                    // being wrapped once per nesting level.
                    Error::NotWellFormed(limit @ WellFormedError::EntityExpansionLimit { .. }) => {
                        Error::from(limit)
                    }
                    other => Error::from(WellFormedError::IllFormedEntity {
                        entity: name.to_owned(),
                        reason: other.to_string(),
                    }),
                })
            }
            None if self.dtd.incomplete => Ok(()),
            None => Err(WellFormedError::UndeclaredEntity(name.to_owned()).into()),
        }
    }

    /// Checks that the replacement text of an entity referenced from content
    /// matches the [`content`] production.
    ///
    /// [`content`]: https://www.w3.org/TR/xml/#NT-content
    fn check_replacement_content(&self, replacement: &str) -> Result<()> {
        let mut nested = self.nested();
        let mut reader = crate::reader::Reader::from_str(replacement);
        loop {
            let event = reader.read_event()?;
            nested.check(&event, Decoder::utf8())?;
            if matches!(event, Event::Eof) {
                break;
            }
        }
        if nested.depth() == 0 {
            Ok(())
        } else {
            Err(
                WellFormedError::MalformedTag("an element is opened but not closed".to_owned())
                    .into(),
            )
        }
    }
}

/// A cursor over the text of a start, end or empty-element tag.
struct TagCursor<'a> {
    /// The text between `<` and `>`, without any trailing `/`.
    input: &'a str,
    /// Byte offset of the cursor inside [`Self::input`].
    pos: usize,
}

impl<'a> TagCursor<'a> {
    /// Whether everything has been consumed.
    #[inline]
    fn is_empty(&self) -> bool {
        self.pos == self.input.len()
    }

    /// A short excerpt of the remaining text, for error messages.
    fn snippet(&self) -> String {
        self.input[self.pos..].chars().take(16).collect()
    }

    /// Consumes zero or more `S` characters, reporting whether any were found.
    fn skip_space(&mut self) -> bool {
        let start = self.pos;
        while matches!(self.input[self.pos..].chars().next(), Some(ch) if is_space(ch)) {
            self.pos += 1;
        }
        self.pos != start
    }

    /// Consumes a `Name` or fails.
    fn name(&mut self) -> Result<&'a str> {
        let start = self.pos;
        match self.input[self.pos..].chars().next() {
            Some(ch) if is_name_start_char(ch) => self.pos += ch.len_utf8(),
            _ => {
                return Err(WellFormedError::InvalidName(self.snippet()).into());
            }
        }
        while let Some(ch) = self.input[self.pos..].chars().next() {
            if is_name_char(ch) {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
        Ok(&self.input[start..self.pos])
    }

    /// Consumes `Eq ::= S? '=' S?`.
    fn eq(&mut self, attribute: &str) -> Result<()> {
        self.skip_space();
        if self.input[self.pos..].starts_with('=') {
            self.pos += 1;
            self.skip_space();
            Ok(())
        } else {
            Err(WellFormedError::MalformedTag(format!(
                "`=` is expected after the attribute name `{}`",
                attribute
            ))
            .into())
        }
    }

    /// Consumes an `AttValue` and returns its content, without the quotes.
    fn att_value(&mut self, attribute: &str) -> Result<&'a str> {
        let quote = match self.input[self.pos..].chars().next() {
            Some(quote @ ('"' | '\'')) => {
                self.pos += 1;
                quote
            }
            _ => {
                return Err(WellFormedError::MalformedTag(format!(
                    "the value of the attribute `{}` must be quoted",
                    attribute
                ))
                .into());
            }
        };
        let body = &self.input[self.pos..];
        let Some(end) = body.find(quote) else {
            return Err(WellFormedError::MalformedTag(format!(
                "the value of the attribute `{}` is not closed",
                attribute
            ))
            .into());
        };
        self.pos += end + 1;
        let value = &body[..end];
        if value.contains('<') {
            return Err(WellFormedError::LessThanInAttributeValue(attribute.to_owned()).into());
        }
        Ok(value)
    }
}

/// Validates a `Name` and produces the matching error when it is malformed.
fn require_name(name: &str) -> Result<()> {
    if is_name(name) {
        Ok(())
    } else {
        Err(WellFormedError::InvalidName(name.to_owned()).into())
    }
}

#[cfg(test)]
mod tests;
