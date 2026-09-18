//! General-entity expansion and attribute-value normalization.
//!
//! Both operations are recursive over entity replacement text, and both are
//! therefore charged against an [`ExpansionLimits`](crate::ExpansionLimits)
//! [`Budget`]: this is where an expansion bomb is stopped.
//!
//! # Examples
//!
//! ```
//! use oxixml_dtd::{DtdParser, ExpansionLimits};
//!
//! let dtd = DtdParser::new().parse_internal_subset(
//!     r#"<!ENTITY inner "world"><!ENTITY outer "hello &inner;">"#,
//! )?;
//!
//! let mut budget = ExpansionLimits::default().budget();
//! assert_eq!(dtd.expand_general_entity("outer", &mut budget)?, "hello world");
//! # Ok::<(), oxixml_dtd::DtdError>(())
//! ```

use oxixml_qname::is_name;

use crate::error::{DtdError, DtdErrorKind};
use crate::limits::Budget;
use crate::model::{Dtd, EntityDefinition};

/// Resolves the body of a character reference — the text between `&#` and `;`.
///
/// Accepts `123` (decimal) and `x1F` (hexadecimal), and rejects anything that
/// is not a legal XML `Char`.
///
/// # Examples
///
/// ```
/// use oxixml_dtd::entity::resolve_char_ref;
///
/// assert_eq!(resolve_char_ref("65"), Some('A'));
/// assert_eq!(resolve_char_ref("x1F6A2"), Some('\u{1F6A2}'));
/// assert_eq!(resolve_char_ref("0"), None); // NUL is not a Char
/// assert_eq!(resolve_char_ref("xZZ"), None);
/// ```
#[must_use]
pub fn resolve_char_ref(body: &str) -> Option<char> {
    let value = match body.strip_prefix('x').or_else(|| body.strip_prefix('X')) {
        Some(hex) if !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit()) => {
            u32::from_str_radix(hex, 16).ok()?
        }
        Some(_) => return None,
        None if !body.is_empty() && body.chars().all(|c| c.is_ascii_digit()) => {
            body.parse::<u32>().ok()?
        }
        None => return None,
    };
    let c = char::from_u32(value)?;
    is_xml_char(c).then_some(c)
}

/// The five entities every conforming XML processor recognizes without a
/// declaration (XML 1.0 §4.6).
#[must_use]
pub fn builtin_entity(name: &str) -> Option<&'static str> {
    match name {
        "amp" => Some("&"),
        "lt" => Some("<"),
        "gt" => Some(">"),
        "apos" => Some("'"),
        "quot" => Some("\""),
        _ => None,
    }
}

/// Builds the WFC "No `<` in Attribute Values" error for the entity chain
/// currently being expanded.
fn less_than_in_attribute_value(stack: &[String]) -> DtdError {
    DtdError::detached(DtdErrorKind::LessThanInAttributeValue {
        chain: stack.to_vec(),
    })
}

/// The XML 1.0 `Char` production `[2]`, minus the surrogates that a Rust
/// [`char`] cannot represent anyway.
#[must_use]
fn is_xml_char(c: char) -> bool {
    matches!(c,
        '\u{9}' | '\u{A}' | '\u{D}'
        | '\u{20}'..='\u{D7FF}'
        | '\u{E000}'..='\u{FFFD}'
        | '\u{10000}'..='\u{10FFFF}')
}

/// How a general-entity reference met during expansion should be treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Attribute-value normalization (XML 1.0 §3.3.3): literal white space
    /// becomes a space, and external entity references are forbidden.
    AttributeValue,
    /// Plain text expansion: white space is preserved.
    Content,
}

impl Dtd {
    /// Expands an internal general entity's replacement text completely,
    /// resolving every nested general-entity and character reference.
    ///
    /// This is the operation an expansion bomb attacks, so every nested
    /// expansion is charged against `budget`.
    ///
    /// # Errors
    ///
    /// * [`DtdErrorKind::UndeclaredGeneralEntity`] when the entity, or one it
    ///   references, is not declared.
    /// * [`DtdErrorKind::UnparsedEntityReference`] when an unparsed entity is
    ///   referenced.
    /// * [`DtdErrorKind::RecursiveEntity`] when the entity references itself.
    /// * [`DtdErrorKind::EntityExpansionLimit`] when `budget` runs out.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_dtd::{DtdParser, ExpansionLimits};
    ///
    /// let dtd = DtdParser::new()
    ///     .parse_internal_subset(r#"<!ENTITY a "A"><!ENTITY b "&a;&a;">"#)?;
    /// let mut budget = ExpansionLimits::default().budget();
    /// assert_eq!(dtd.expand_general_entity("b", &mut budget)?, "AA");
    /// # Ok::<(), oxixml_dtd::DtdError>(())
    /// ```
    pub fn expand_general_entity(
        &self,
        name: &str,
        budget: &mut Budget,
    ) -> Result<String, DtdError> {
        let mut out = String::new();
        let mut stack = Vec::new();
        self.include_entity(name, &mut out, budget, &mut stack, Mode::Content)?;
        Ok(out)
    }

    /// Normalizes an attribute value literal per XML 1.0 §3.3.3.
    ///
    /// `literal` is the text between the quotes, with line ends already
    /// normalized. `tokenized` selects the extra collapsing that applies to
    /// every attribute type except `CDATA`.
    ///
    /// # Errors
    ///
    /// The same errors as [`expand_general_entity`](Self::expand_general_entity),
    /// plus [`DtdErrorKind::InvalidCharRef`] for a malformed character
    /// reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_dtd::{Dtd, ExpansionLimits};
    ///
    /// let dtd = Dtd::new();
    /// let mut budget = ExpansionLimits::default().budget();
    ///
    /// // CDATA: white space becomes single spaces but is not collapsed.
    /// assert_eq!(dtd.normalize_attribute_value("  a\tb  ", false, &mut budget)?, "  a b  ");
    /// // Tokenized: runs collapse and the ends are trimmed.
    /// assert_eq!(dtd.normalize_attribute_value("  a\tb  ", true, &mut budget)?, "a b");
    /// // A character reference is not subject to the white-space rule, so a
    /// // referenced tab survives where a literal one would have become a space.
    /// assert_eq!(dtd.normalize_attribute_value("&#9;a", true, &mut budget)?, "\ta");
    /// # Ok::<(), oxixml_dtd::DtdError>(())
    /// ```
    pub fn normalize_attribute_value(
        &self,
        literal: &str,
        tokenized: bool,
        budget: &mut Budget,
    ) -> Result<String, DtdError> {
        let mut out = String::new();
        let mut stack = Vec::new();
        self.append_normalized(literal, &mut out, budget, &mut stack, Mode::AttributeValue)?;
        if !tokenized {
            return Ok(out);
        }
        let collapsed: Vec<&str> = out.split(' ').filter(|part| !part.is_empty()).collect();
        Ok(collapsed.join(" "))
    }

    /// Appends `literal`, expanding references, to `out`.
    fn append_normalized(
        &self,
        literal: &str,
        out: &mut String,
        budget: &mut Budget,
        stack: &mut Vec<String>,
        mode: Mode,
    ) -> Result<(), DtdError> {
        let mut rest = literal;
        while let Some(index) = rest.find('&') {
            self.append_text(&rest[..index], out, mode, stack)?;
            let after = &rest[index + 1..];
            let Some(end) = after.find(';') else {
                out.push('&');
                rest = after;
                continue;
            };
            let body = &after[..end];
            rest = &after[end + 1..];
            if let Some(char_body) = body.strip_prefix('#') {
                let Some(c) = resolve_char_ref(char_body) else {
                    return Err(DtdError::detached(DtdErrorKind::InvalidCharRef(
                        char_body.to_owned(),
                    )));
                };
                // WFC: No `<` in Attribute Values (§3.1). A character reference
                // written *in the attribute value itself* is fine — it is not
                // markup — but the replacement text of an entity has its
                // character references already resolved (§4.5), so a `<` that
                // arrives from inside an expansion breaks the constraint.
                if mode == Mode::AttributeValue && c == '<' && !stack.is_empty() {
                    return Err(less_than_in_attribute_value(stack));
                }
                out.push(c);
            } else if is_name(body) {
                self.include_entity(body, out, budget, stack, mode)?;
            } else {
                out.push('&');
                self.append_text(body, out, mode, stack)?;
                out.push(';');
            }
        }
        self.append_text(rest, out, mode, stack)?;
        Ok(())
    }

    /// Appends literal text, applying the white-space rule of the current mode.
    ///
    /// # Errors
    ///
    /// [`DtdErrorKind::LessThanInAttributeValue`] when a `<` appears in the
    /// replacement text of an entity being expanded into an attribute value
    /// (`stack` non-empty).
    fn append_text(
        &self,
        text: &str,
        out: &mut String,
        mode: Mode,
        stack: &[String],
    ) -> Result<(), DtdError> {
        match mode {
            Mode::Content => out.push_str(text),
            Mode::AttributeValue => {
                for c in text.chars() {
                    if matches!(c, ' ' | '\t' | '\n' | '\r') {
                        out.push(' ');
                    } else {
                        if c == '<' && !stack.is_empty() {
                            return Err(less_than_in_attribute_value(stack));
                        }
                        out.push(c);
                    }
                }
            }
        }
        Ok(())
    }

    /// Expands one general entity into `out`.
    fn include_entity(
        &self,
        name: &str,
        out: &mut String,
        budget: &mut Budget,
        stack: &mut Vec<String>,
        mode: Mode,
    ) -> Result<(), DtdError> {
        if let Some(builtin) = builtin_entity(name) {
            // The five entities of XML 1.0 §4.6 need no declaration. `&amp;`
            // expands to a literal `&`, which must *not* start a new reference.
            out.push_str(builtin);
            return Ok(());
        }
        if stack.iter().any(|entry| entry == name) {
            return Err(DtdError::detached(DtdErrorKind::RecursiveEntity {
                name: name.to_owned(),
                chain: stack.clone(),
            }));
        }
        let Some(entity) = self.general_entity(name) else {
            return Err(DtdError::detached(DtdErrorKind::UndeclaredGeneralEntity(
                name.to_owned(),
            )));
        };
        let text = match entity.definition() {
            EntityDefinition::Internal(text) => text,
            EntityDefinition::External {
                notation: Some(_), ..
            } => {
                return Err(DtdError::detached(DtdErrorKind::UnparsedEntityReference(
                    name.to_owned(),
                )));
            }
            EntityDefinition::External { id, notation: None } => {
                // WFC: No External Entity References (XML 1.0 §3.1) forbids
                // this in an attribute value; elsewhere the caller must supply
                // the text through a resolver instead of coming through here.
                return Err(DtdError::detached(DtdErrorKind::ExternalEntityDenied {
                    system_id: id.system_id().unwrap_or_default().to_owned(),
                }));
            }
        };
        budget.check_depth(stack.len() + 1)?;
        budget.charge_expansion(text.chars().count())?;
        stack.push(name.to_owned());
        let result = self.append_normalized(text, out, budget, stack, mode);
        stack.pop();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::ExpansionLimits;
    use crate::model::{EntityDecl, EntityKind};

    fn dtd_with(entities: &[(&str, &str)]) -> Dtd {
        let mut dtd = Dtd::new();
        for (name, text) in entities {
            dtd.add_entity(EntityDecl::new(
                *name,
                EntityKind::General,
                EntityDefinition::Internal((*text).to_owned()),
                false,
            ));
        }
        dtd
    }

    #[test]
    fn character_references_cover_both_radices() {
        assert_eq!(resolve_char_ref("10"), Some('\n'));
        assert_eq!(resolve_char_ref("xA"), Some('\n'));
        assert_eq!(resolve_char_ref("XA"), Some('\n'));
        assert_eq!(resolve_char_ref(""), None);
        assert_eq!(resolve_char_ref("x"), None);
        assert_eq!(resolve_char_ref("1114112"), None);
        assert_eq!(resolve_char_ref("8"), None);
    }

    #[test]
    fn nested_entities_expand_depth_first() {
        let dtd = dtd_with(&[("a", "A"), ("b", "&a;&a;"), ("c", "[&b;]")]);
        let mut budget = ExpansionLimits::default().budget();
        assert_eq!(
            dtd.expand_general_entity("c", &mut budget).as_deref(),
            Ok("[AA]")
        );
    }

    #[test]
    fn self_reference_is_reported_as_recursion() {
        let dtd = dtd_with(&[("a", "&a;")]);
        let mut budget = ExpansionLimits::default().budget();
        let error = dtd
            .expand_general_entity("a", &mut budget)
            .expect_err("recursive");
        assert!(matches!(error.kind(), DtdErrorKind::RecursiveEntity { .. }));
    }

    #[test]
    fn undeclared_entities_are_reported() {
        let dtd = dtd_with(&[]);
        let mut budget = ExpansionLimits::default().budget();
        let error = dtd
            .expand_general_entity("nope", &mut budget)
            .expect_err("undeclared");
        assert!(matches!(
            error.kind(),
            DtdErrorKind::UndeclaredGeneralEntity(_)
        ));
    }

    #[test]
    fn attribute_normalization_collapses_only_for_tokenized_types() {
        let dtd = dtd_with(&[]);
        let mut budget = ExpansionLimits::default().budget();
        assert_eq!(
            dtd.normalize_attribute_value("a\r\n b", false, &mut budget)
                .as_deref(),
            Ok("a   b")
        );
        assert_eq!(
            dtd.normalize_attribute_value("a\r\n b", true, &mut budget)
                .as_deref(),
            Ok("a b")
        );
    }

    #[test]
    fn entity_text_is_normalized_recursively_in_attribute_values() {
        let dtd = dtd_with(&[("ws", "x\ty")]);
        let mut budget = ExpansionLimits::default().budget();
        assert_eq!(
            dtd.normalize_attribute_value("&ws;", false, &mut budget)
                .as_deref(),
            Ok("x y")
        );
    }

    #[test]
    fn an_unresolvable_ampersand_is_left_alone() {
        let dtd = dtd_with(&[]);
        let mut budget = ExpansionLimits::default().budget();
        assert_eq!(
            dtd.normalize_attribute_value("a & b", false, &mut budget)
                .as_deref(),
            Ok("a & b")
        );
    }
}
