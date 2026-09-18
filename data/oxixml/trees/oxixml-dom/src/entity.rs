//! Expansion of the general entities declared in a document's internal subset.
//!
//! [XML 1.0 §4.5](https://www.w3.org/TR/xml/#intern-replacement) distinguishes
//! two forms of an internal entity's value. The *literal entity value* is the
//! quoted string as it is written in the declaration; the *replacement text* is
//! what a reference to the entity actually stands for. They are related by a
//! single left-to-right pass over the literal in which character references are
//! replaced by the character they denote ("Included") while general entity
//! references are kept exactly as written ("Bypassed"). [`replacement_text`]
//! performs that pass, and it is the *replacement text* that is stored.
//!
//! A `&name;` in content is then expanded by a second single pass over the
//! stored replacement text ([XML 1.0 §4.4.2](https://www.w3.org/TR/xml/#included)),
//! in which character references are replaced and nested general entity
//! references are resolved recursively. Neither pass ever rescans what it has
//! just produced, which is what makes the specification's doubled `&#38;#38;`
//! idiom come out right: the declaration `<!ENTITY b "&#38;#60;">` has the
//! replacement text `&#60;`, and `&b;` in content is the single character `<`.
//!
//! # What is not expanded
//!
//! The replacement text is included as *character data*. A `<` that survives
//! into the replacement text starts markup, which cannot be spliced into a tree
//! that is already being built — this is exactly why
//! [XML 1.0 §4.6](https://www.w3.org/TR/xml/#sec-predefined-ent) requires `lt`
//! and `amp` to be declared with a *doubly* escaped value. Such a reference is
//! reported as [`ExpandError::Markup`] and left to the caller, which keeps it as
//! an unexpanded entity reference rather than turning markup into text.
//!
//! # Limits
//!
//! A self- or mutually-recursive entity is a well-formedness violation
//! ([XML 1.0 WFC: No Recursion](https://www.w3.org/TR/xml/#norecursion)) and is
//! reported rather than followed. Entities that merely nest deeply, or that
//! multiply their own replacement text ("billion laughs"), are bounded by
//! [`MAX_ENTITY_DEPTH`], [`MAX_NESTED_EXPANSIONS`] and [`MAX_EXPANDED_BYTES`].
//!
//! What these budgets bound is *amplification*, not document size. A reference
//! written in the document itself simply stands for its own replacement text,
//! so it costs nothing but the characters it produces, however often it is
//! repeated: a page with fifty thousand `&nbsp;` in it is an ordinary document,
//! not an attack. Only an expansion that happens *inside* another entity's
//! replacement text multiplies what the document asked for, and only those are
//! counted.
//!
//! These rules, the limits and the resulting character data are shared verbatim
//! with `oxixml-sax` and `oxixml-infoset`, so a document that declares entities
//! means the same thing whichever of the three reads it.

use std::collections::HashMap;
use std::fmt;

use oxixml_xml::XmlVersion;
use oxixml_xml::escape::resolve_xml_entity;

use crate::names::is_char;

/// The maximum depth to which nested general entity references are expanded,
/// both inside an attribute value and inside character data.
pub(crate) const MAX_ENTITY_DEPTH: usize = 16;

/// The maximum number of *nested* general entity expansions — those performed
/// inside another entity's replacement text — while building one document.
///
/// References written in the document itself are not counted: they are bounded
/// by the size of the document, and by [`MAX_EXPANDED_BYTES`].
pub(crate) const MAX_NESTED_EXPANSIONS: usize = 100_000;

/// The maximum total size, in bytes, of the character data produced by expanding
/// entity references while building one document.
pub(crate) const MAX_EXPANDED_BYTES: usize = 10 * 1024 * 1024;

/// Which expansion limit was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Limit {
    /// [`MAX_ENTITY_DEPTH`].
    Depth,
    /// [`MAX_NESTED_EXPANSIONS`].
    Expansions,
    /// [`MAX_EXPANDED_BYTES`].
    Bytes,
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Depth => write!(f, "the maximum entity nesting depth of {MAX_ENTITY_DEPTH}"),
            Self::Expansions => write!(
                f,
                "the maximum of {MAX_NESTED_EXPANSIONS} nested entity expansions per document"
            ),
            Self::Bytes => write!(
                f,
                "the maximum of {MAX_EXPANDED_BYTES} bytes of expanded entity text per document"
            ),
        }
    }
}

/// Why a general entity reference could not be expanded into character data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExpandError {
    /// No declaration for the entity was found in the internal subset. It may be
    /// external, declared in an external subset this crate never reads, or
    /// simply missing.
    Undeclared {
        /// The name of the entity.
        name: String,
    },
    /// The entity's replacement text contains a `<`, which begins markup rather
    /// than character data.
    Markup {
        /// The name of the entity.
        name: String,
    },
    /// The entity refers to itself, directly or through other entities.
    Recursion {
        /// The name of the entity that closed the cycle.
        name: String,
    },
    /// An expansion limit was reached.
    Limit {
        /// The name of the entity whose expansion was refused.
        name: String,
        /// The limit that was reached.
        limit: Limit,
    },
}

impl ExpandError {
    /// Returns the name of the entity the error is about.
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Undeclared { name }
            | Self::Markup { name }
            | Self::Recursion { name }
            | Self::Limit { name, .. } => name,
        }
    }
}

impl fmt::Display for ExpandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Undeclared { name } => {
                write!(f, "the general entity `&{name};` is not declared")
            }
            Self::Markup { name } => write!(
                f,
                "the replacement text of the general entity `&{name};` contains markup, which \
                 cannot be included as character data"
            ),
            Self::Recursion { name } => write!(
                f,
                "the general entity `&{name};` refers to itself, which XML 1.0 forbids"
            ),
            Self::Limit { name, limit } => write!(
                f,
                "expanding the general entity `&{name};` would exceed {limit}"
            ),
        }
    }
}

/// Builds the replacement text of an internal general entity from its literal
/// entity value — the text between the quotes of an `<!ENTITY name "…">`
/// declaration.
///
/// Character references are replaced by the character they denote; every other
/// `&…;` is kept exactly as written, to be resolved when the entity is used.
/// `version` selects which characters a reference is allowed to denote — XML 1.1
/// admits the C0 controls, XML 1.0 does not.
pub(crate) fn replacement_text(literal: &str, version: XmlVersion) -> String {
    if !literal.contains("&#") {
        return literal.to_owned();
    }
    let mut out = String::with_capacity(literal.len());
    let mut rest = literal;
    while let Some(offset) = rest.find('&') {
        out.push_str(&rest[..offset]);
        let reference = &rest[offset..];
        match char_reference(reference, version) {
            Some((character, length)) => {
                out.push(character);
                rest = &reference[length..];
            }
            None => {
                out.push('&');
                rest = &reference['&'.len_utf8()..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Expands general entity references on behalf of one document.
///
/// The expansion budget is carried here rather than on each call so that a
/// document made of many small but self-multiplying references is bounded as a
/// whole, not one reference at a time.
#[derive(Debug, Default)]
pub(crate) struct Expander {
    /// Nested expansions performed so far.
    expansions: usize,
    /// Bytes of expanded character data produced so far.
    bytes: usize,
    /// The version whose `Char` production the character references inside a
    /// replacement text are held to.
    version: XmlVersion,
}

impl Expander {
    /// Selects the XML version the expander resolves character references
    /// under. Called once the XML declaration has been seen.
    pub(crate) const fn set_version(&mut self, version: XmlVersion) {
        self.version = version;
    }

    /// Expands the reference `&name;` appearing in content into the character
    /// data it stands for.
    ///
    /// # Errors
    ///
    /// Returns [`ExpandError`] when the entity is not declared in `entities`,
    /// when its replacement text cannot be included as character data, when it
    /// is recursive, or when an expansion limit is reached.
    pub(crate) fn expand(
        &mut self,
        name: &str,
        entities: &HashMap<String, String>,
    ) -> Result<String, ExpandError> {
        let mut open = Vec::new();
        self.include(name, entities, &mut open)
    }

    /// Performs the "Included" step of XML 1.0 §4.4.2 for one entity.
    fn include(
        &mut self,
        name: &str,
        entities: &HashMap<String, String>,
        open: &mut Vec<String>,
    ) -> Result<String, ExpandError> {
        let Some(text) = entities.get(name) else {
            return Err(ExpandError::Undeclared {
                name: name.to_owned(),
            });
        };
        if text.contains('<') {
            return Err(ExpandError::Markup {
                name: name.to_owned(),
            });
        }
        if open.iter().any(|open_name| open_name == name) {
            return Err(ExpandError::Recursion {
                name: name.to_owned(),
            });
        }
        if open.len() >= MAX_ENTITY_DEPTH {
            return Err(ExpandError::Limit {
                name: name.to_owned(),
                limit: Limit::Depth,
            });
        }
        // Only an expansion *inside* another entity's replacement text multiplies
        // the document; a reference the document itself contains costs no more
        // than the characters it stands for.
        if !open.is_empty() {
            self.charge_expansion(name)?;
        }

        open.push(name.to_owned());
        let expanded = self.resolve(text, entities, open);
        open.pop();
        let expanded = expanded?;
        // Charging what was *produced* is what bounds an entity whose replacement
        // text is short but multiplies: its own bytes stay tiny while its output
        // grows exponentially.
        self.charge_bytes(name, expanded.len())?;
        Ok(expanded)
    }

    /// Resolves every reference in one entity's replacement text, in a single
    /// left-to-right pass that never revisits what it has produced.
    fn resolve(
        &mut self,
        text: &str,
        entities: &HashMap<String, String>,
        open: &mut Vec<String>,
    ) -> Result<String, ExpandError> {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(offset) = rest.find('&') {
            out.push_str(&rest[..offset]);
            let reference = &rest[offset..];
            if let Some((character, length)) = char_reference(reference, self.version) {
                out.push(character);
                rest = &reference[length..];
                continue;
            }
            match entity_reference(reference) {
                Some((name, length)) => {
                    match resolve_xml_entity(name) {
                        Some(predefined) => out.push_str(predefined),
                        None => out.push_str(&self.include(name, entities, open)?),
                    }
                    rest = &reference[length..];
                }
                // A bare `&` that begins no reference is data, not markup: it can
                // only have reached the replacement text through a `&#38;` in the
                // declaration, which is how a document writes an ampersand that
                // is meant to stay one.
                None => {
                    out.push('&');
                    rest = &reference['&'.len_utf8()..];
                }
            }
        }
        out.push_str(rest);
        Ok(out)
    }

    /// Charges one expansion of `name`, whose replacement text is `len` bytes
    /// long, against the document's budget, and reports whether it fits.
    ///
    /// Attribute-value normalization is performed by [`oxixml_xml`], which walks
    /// the nested references itself and is bounded only by
    /// [`MAX_ENTITY_DEPTH`]; a *wide* declaration multiplies without ever
    /// nesting deeply, so the budget has to be applied where the replacement
    /// text is handed over. One call per occurrence, including every occurrence
    /// reached recursively, makes the count of calls the amount of amplification.
    pub(crate) fn charge_attribute_expansion(&mut self, name: &str, len: usize) -> bool {
        self.charge_expansion(name).is_ok() && self.charge_bytes(name, len).is_ok()
    }

    /// Charges one nested expansion of `name` against the document's budget.
    fn charge_expansion(&mut self, name: &str) -> Result<(), ExpandError> {
        self.expansions = self.expansions.saturating_add(1);
        if self.expansions > MAX_NESTED_EXPANSIONS {
            return Err(ExpandError::Limit {
                name: name.to_owned(),
                limit: Limit::Expansions,
            });
        }
        Ok(())
    }

    /// Charges the text of a retrieved external parsed general entity against
    /// the same budget as an internal expansion, so that a chain of external
    /// entities cannot outgrow what one document is allowed to produce.
    pub(crate) fn charge_external_entity(
        &mut self,
        name: &str,
        len: usize,
    ) -> Result<(), ExpandError> {
        self.charge_expansion(name)?;
        self.charge_bytes(name, len)
    }

    /// Charges `len` bytes of expanded text against the document's budget.
    fn charge_bytes(&mut self, name: &str, len: usize) -> Result<(), ExpandError> {
        self.bytes = self.bytes.saturating_add(len);
        if self.bytes > MAX_EXPANDED_BYTES {
            return Err(ExpandError::Limit {
                name: name.to_owned(),
                limit: Limit::Bytes,
            });
        }
        Ok(())
    }
}

/// Parses a character reference (`&#N;` or `&#xN;`) at the start of `text`,
/// returning the character it denotes and the length of the reference in bytes.
///
/// Returns `None` when `text` does not begin with a syntactically valid
/// reference to a legal XML character, in which case the leading `&` is data.
fn char_reference(text: &str, version: XmlVersion) -> Option<(char, usize)> {
    let body = text.strip_prefix("&#")?;
    let end = body.find(';')?;
    let digits = &body[..end];
    let code = match digits.strip_prefix('x') {
        // `CharRef ::= '&#' [0-9]+ ';' | '&#x' [0-9a-fA-F]+ ';'` — the `x` is
        // lower case, but the digits after it may be either case.
        Some(hex) if !hex.is_empty() && hex.bytes().all(|byte| byte.is_ascii_hexdigit()) => {
            u32::from_str_radix(hex, 16).ok()?
        }
        Some(_) => return None,
        None if !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) => {
            digits.parse::<u32>().ok()?
        }
        None => return None,
    };
    let character = char::from_u32(code)?;
    is_char(character, version).then_some((character, "&#".len() + end + ';'.len_utf8()))
}

/// Parses a general entity reference (`&name;`) at the start of `text`,
/// returning the name and the length of the reference in bytes.
///
/// The name is only checked for the delimiters that could not appear in a
/// `Name`; anything else is looked up and reported as undeclared if it is not
/// there, which is the same outcome a stricter check would produce.
fn entity_reference(text: &str) -> Option<(&str, usize)> {
    let body = text.strip_prefix('&')?;
    let end = body.find(';')?;
    let name = &body[..end];
    let plausible = !name.is_empty()
        && !name.starts_with('#')
        && !name.chars().any(|character| {
            character.is_whitespace() || matches!(character, '&' | '<' | '>' | '"' | '\'' | '%')
        });
    plausible.then_some((name, '&'.len_utf8() + end + ';'.len_utf8()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entities(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|&(name, literal)| {
                (
                    name.to_owned(),
                    replacement_text(literal, XmlVersion::Implicit1_0),
                )
            })
            .collect()
    }

    #[test]
    fn character_references_are_included_at_declaration_time() {
        assert_eq!(
            replacement_text("&#169;", XmlVersion::Implicit1_0),
            "\u{a9}"
        );
        assert_eq!(
            replacement_text("&#xa9;", XmlVersion::Implicit1_0),
            "\u{a9}"
        );
        assert_eq!(
            replacement_text("bar &amp; baz", XmlVersion::Implicit1_0),
            "bar &amp; baz"
        );
        // The doubled form of the specification's own example: only the outer
        // reference is resolved, and what it produces is not rescanned.
        assert_eq!(
            replacement_text("&#38;#60;", XmlVersion::Implicit1_0),
            "&#60;"
        );
        assert_eq!(
            replacement_text("&#38;#38;#38;", XmlVersion::Implicit1_0),
            "&#38;#38;"
        );
        assert_eq!(
            replacement_text("&#38;rights;", XmlVersion::Implicit1_0),
            "&rights;"
        );
    }

    #[test]
    fn malformed_character_references_stay_literal() {
        assert_eq!(replacement_text("&#zz;", XmlVersion::Implicit1_0), "&#zz;");
        assert_eq!(
            replacement_text("&#X41;", XmlVersion::Implicit1_0),
            "&#X41;"
        );
        assert_eq!(replacement_text("&#;", XmlVersion::Implicit1_0), "&#;");
        assert_eq!(replacement_text("&#38", XmlVersion::Implicit1_0), "&#38");
        // U+0000 and the surrogates are not `Char`s.
        assert_eq!(replacement_text("&#0;", XmlVersion::Implicit1_0), "&#0;");
        assert_eq!(
            replacement_text("&#xD800;", XmlVersion::Implicit1_0),
            "&#xD800;"
        );
    }

    #[test]
    fn xml11_character_references_denote_the_c0_controls() {
        // XML 1.1 widens `Char` to `[#x1-#xD7FF] | …`, so a C0 control named by
        // a character reference is included in the replacement text; under 1.0
        // the same reference is not a reference at all and stays literal.
        assert_eq!(replacement_text("&#x7;", XmlVersion::Explicit1_1), "\u{7}");
        assert_eq!(replacement_text("&#x7;", XmlVersion::Implicit1_0), "&#x7;");
        // `#x0` is outside `Char` in both versions.
        assert_eq!(replacement_text("&#0;", XmlVersion::Explicit1_1), "&#0;");
    }

    #[test]
    fn nested_references_are_expanded_at_use_time() {
        let declared = entities(&[
            ("amp_then_lt", "&#38;#60;"),
            ("thrice", "&#38;#38;#38;"),
            ("rights", "All rights reserved"),
            (
                "book",
                "La Peste: Les Editions de Minuit, &#38;#38; 1947, &#38;rights;",
            ),
            ("outer", "[&inner;]"),
            ("inner", "&#169;"),
        ]);
        let mut expander = Expander::default();
        assert_eq!(
            expander.expand("amp_then_lt", &declared),
            Ok("<".to_owned())
        );
        assert_eq!(expander.expand("thrice", &declared), Ok("&#38;".to_owned()));
        assert_eq!(
            expander.expand("book", &declared),
            Ok("La Peste: Les Editions de Minuit, & 1947, All rights reserved".to_owned())
        );
        assert_eq!(
            expander.expand("outer", &declared),
            Ok("[\u{a9}]".to_owned())
        );
    }

    #[test]
    fn predefined_entities_are_resolved_without_being_rescanned() {
        let declared = entities(&[("tag", "&lt;foo&gt;"), ("double", "&amp;amp;")]);
        let mut expander = Expander::default();
        assert_eq!(expander.expand("tag", &declared), Ok("<foo>".to_owned()));
        assert_eq!(expander.expand("double", &declared), Ok("&amp;".to_owned()));
    }

    #[test]
    fn markup_in_the_replacement_text_is_reported() {
        let declared = entities(&[("para", "<p>x</p>"), ("bare", "&#60;")]);
        let mut expander = Expander::default();
        assert_eq!(
            expander.expand("para", &declared),
            Err(ExpandError::Markup {
                name: "para".to_owned()
            })
        );
        // A single (rather than doubled) escape of `<` also leaves markup behind.
        assert_eq!(
            expander.expand("bare", &declared),
            Err(ExpandError::Markup {
                name: "bare".to_owned()
            })
        );
    }

    #[test]
    fn recursive_entities_are_rejected_rather_than_followed() {
        let declared = entities(&[("self", "&self;"), ("a", "&b;"), ("b", "&a;")]);
        let mut expander = Expander::default();
        assert_eq!(
            expander.expand("self", &declared),
            Err(ExpandError::Recursion {
                name: "self".to_owned()
            })
        );
        assert_eq!(
            expander.expand("a", &declared),
            Err(ExpandError::Recursion {
                name: "a".to_owned()
            })
        );
    }

    #[test]
    fn deep_nesting_is_bounded() {
        let mut declared = HashMap::new();
        for level in 0..=MAX_ENTITY_DEPTH + 4 {
            declared.insert(format!("e{level}"), format!("&e{};", level + 1));
        }
        declared.insert(format!("e{}", MAX_ENTITY_DEPTH + 5), "leaf".to_owned());
        let mut expander = Expander::default();
        assert_eq!(
            expander.expand("e0", &declared),
            Err(ExpandError::Limit {
                name: format!("e{MAX_ENTITY_DEPTH}"),
                limit: Limit::Depth,
            })
        );
    }

    #[test]
    fn a_document_may_repeat_a_reference_as_often_as_it_likes() {
        // Repetition is not amplification: an ordinary page can hold tens of
        // thousands of `&nbsp;`, and every one of them must expand.
        let declared = entities(&[("nbsp", "&#160;")]);
        let mut expander = Expander::default();
        for _ in 0..MAX_NESTED_EXPANSIONS + 1_000 {
            assert_eq!(expander.expand("nbsp", &declared), Ok("\u{a0}".to_owned()));
        }
    }

    #[test]
    fn repetition_is_still_bounded_by_what_it_produces() {
        // …but only while what it produces stays within the document's budget.
        let declared = entities(&[("big", &"x".repeat(64 * 1024))]);
        let mut expander = Expander::default();
        let error = loop {
            match expander.expand("big", &declared) {
                Ok(_) => {}
                Err(error) => break error,
            }
        };
        assert!(
            matches!(
                error,
                ExpandError::Limit {
                    limit: Limit::Bytes,
                    ..
                }
            ),
            "{error}"
        );
    }

    #[test]
    fn multiplying_entities_are_bounded() {
        // The classic "billion laughs": each level names the previous one ten
        // times, so the expanded size grows exponentially while the depth stays
        // small enough to pass the depth check.
        let mut declared = HashMap::new();
        declared.insert("lol0".to_owned(), "lol".repeat(10));
        for level in 1..10 {
            declared.insert(
                format!("lol{level}"),
                format!("&lol{};", level - 1).repeat(10),
            );
        }
        let mut expander = Expander::default();
        let error = expander
            .expand("lol9", &declared)
            .expect_err("a self-multiplying entity must be refused");
        assert!(matches!(error, ExpandError::Limit { .. }), "{error}");
    }

    #[test]
    fn undeclared_entities_are_reported_with_their_name() {
        let declared = entities(&[("known", "x&missing;y")]);
        let mut expander = Expander::default();
        assert_eq!(
            expander.expand("known", &declared),
            Err(ExpandError::Undeclared {
                name: "missing".to_owned()
            })
        );
        assert_eq!(
            expander
                .expand("absent", &declared)
                .expect_err("an undeclared entity has no replacement text")
                .name(),
            "absent"
        );
    }

    #[test]
    fn a_bare_ampersand_in_the_replacement_text_is_data() {
        let declared = entities(&[("company", "AT&#38;T"), ("stray", "&#38; ")]);
        let mut expander = Expander::default();
        assert_eq!(expander.expand("company", &declared), Ok("AT&T".to_owned()));
        assert_eq!(expander.expand("stray", &declared), Ok("& ".to_owned()));
    }
}
