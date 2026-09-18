//! XML character-class productions.
//!
//! These predicates come directly from the grammar productions of the XML
//! specifications, **not** from the Unicode Character Database, and so they are
//! deliberately hand-written from the spec rather than table-generated:
//!
//! * [XML 1.0, Fifth Edition][xml10] — the `Char` production `[2]`, the
//!   `NameStartChar` production `[4]`, the `NameChar` production `[4a]`, and the
//!   white-space production `S` `[3]`.
//! * [XML 1.1][xml11] — the `Char` production `[2]`, the `RestrictedChar`
//!   production `[2a]`, and the same `NameStartChar` / `NameChar` productions
//!   (XML 1.0 Fifth Edition adopted the XML 1.1 name grammar, so the name
//!   classes are identical between the two versions).
//!
//! Rust's [`char`] can never be a UTF-16 surrogate (`U+D800..=U+DFFF`), so the
//! surrogate exclusions in the `Char` productions are enforced by the type
//! system and are not repeated here.
//!
//! [xml10]: https://www.w3.org/TR/2008/REC-xml-20081126/
//! [xml11]: https://www.w3.org/TR/2006/REC-xml11-20060816/

/// A version of the XML specification, selecting the `Char` and
/// `RestrictedChar` productions to apply.
///
/// The `NameStartChar` and `NameChar` productions are identical between the two
/// versions, so [`is_name_start_char`] and [`is_name_char`] are not
/// parameterised by version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum XmlVersion {
    /// XML 1.0 (Fifth Edition). The `Char` production forbids the C0 control
    /// characters other than tab, line feed, and carriage return.
    #[default]
    V10,
    /// XML 1.1. The `Char` production admits every C0 control except `NUL`;
    /// the disallowed-in-source controls are instead classified as *restricted*
    /// (see [`is_restricted_char`]).
    V11,
}

/// Returns `true` if `c` matches the XML `Char` production for `version`.
///
/// This dispatches to [`is_char_v10`] or [`is_char_v11`].
///
/// # Examples
///
/// ```
/// use oxixml_unicode::xml::{is_char, XmlVersion};
///
/// // U+0001 is legal content in XML 1.1 but not in XML 1.0.
/// assert!(!is_char('\u{1}', XmlVersion::V10));
/// assert!(is_char('\u{1}', XmlVersion::V11));
/// ```
#[must_use]
pub fn is_char(c: char, version: XmlVersion) -> bool {
    match version {
        XmlVersion::V10 => is_char_v10(c),
        XmlVersion::V11 => is_char_v11(c),
    }
}

/// Returns `true` if `c` matches the XML 1.0 (Fifth Edition) `Char` production:
///
/// ```text
/// Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
/// ```
///
/// # Examples
///
/// ```
/// use oxixml_unicode::xml::is_char_v10;
///
/// assert!(is_char_v10('\t'));
/// assert!(is_char_v10('a'));
/// assert!(!is_char_v10('\u{0}'));   // NUL
/// assert!(!is_char_v10('\u{8}'));   // backspace
/// assert!(!is_char_v10('\u{FFFF}')); // noncharacter
/// ```
#[must_use]
pub fn is_char_v10(c: char) -> bool {
    matches!(c,
        '\u{9}' | '\u{A}' | '\u{D}'
        | '\u{20}'..='\u{D7FF}'
        | '\u{E000}'..='\u{FFFD}'
        | '\u{10000}'..='\u{10FFFF}')
}

/// Returns `true` if `c` matches the XML 1.1 `Char` production:
///
/// ```text
/// Char ::= [#x1-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
/// ```
///
/// Every code point except `NUL` (and the surrogates / `U+FFFE` / `U+FFFF`) is
/// a legal `Char`; those that are nonetheless discouraged in source text are
/// the *restricted* characters (see [`is_restricted_char`]).
///
/// # Examples
///
/// ```
/// use oxixml_unicode::xml::is_char_v11;
///
/// assert!(is_char_v11('\u{1}'));
/// assert!(!is_char_v11('\u{0}'));   // NUL is never a Char
/// assert!(!is_char_v11('\u{FFFE}'));
/// ```
#[must_use]
pub fn is_char_v11(c: char) -> bool {
    matches!(c,
        '\u{1}'..='\u{D7FF}'
        | '\u{E000}'..='\u{FFFD}'
        | '\u{10000}'..='\u{10FFFF}')
}

/// Returns `true` if `c` matches the XML 1.1 `RestrictedChar` production:
///
/// ```text
/// RestrictedChar ::= [#x1-#x8] | [#xB-#xC] | [#xE-#x1F] | [#x7F-#x84] | [#x86-#x9F]
/// ```
///
/// Restricted characters are legal XML 1.1 [`Char`](is_char_v11)s but must be
/// written as character references when they appear in element content or
/// attribute values.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::xml::is_restricted_char;
///
/// assert!(is_restricted_char('\u{1}'));
/// assert!(is_restricted_char('\u{7F}'));
/// assert!(!is_restricted_char('\u{9}')); // tab is ordinary white space
/// assert!(!is_restricted_char('a'));
/// ```
#[must_use]
pub fn is_restricted_char(c: char) -> bool {
    matches!(c,
        '\u{1}'..='\u{8}'
        | '\u{B}'..='\u{C}'
        | '\u{E}'..='\u{1F}'
        | '\u{7F}'..='\u{84}'
        | '\u{86}'..='\u{9F}')
}

/// Returns `true` if `c` matches the XML `NameStartChar` production `[4]`:
///
/// ```text
/// NameStartChar ::= ":" | [A-Z] | "_" | [a-z] | [#xC0-#xD6] | [#xD8-#xF6]
///                 | [#xF8-#x2FF] | [#x370-#x37D] | [#x37F-#x1FFF]
///                 | [#x200C-#x200D] | [#x2070-#x218F] | [#x2C00-#x2FEF]
///                 | [#x3001-#xD7FF] | [#xF900-#xFDCF] | [#xFDF0-#xFFFD]
///                 | [#x10000-#xEFFFF]
/// ```
///
/// This production is identical in XML 1.0 (Fifth Edition) and XML 1.1.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::xml::is_name_start_char;
///
/// assert!(is_name_start_char(':'));
/// assert!(is_name_start_char('_'));
/// assert!(is_name_start_char('é'));
/// assert!(!is_name_start_char('-'));
/// assert!(!is_name_start_char('0'));
/// ```
#[must_use]
pub fn is_name_start_char(c: char) -> bool {
    matches!(c,
        ':' | 'A'..='Z' | '_' | 'a'..='z'
        | '\u{C0}'..='\u{D6}'
        | '\u{D8}'..='\u{F6}'
        | '\u{F8}'..='\u{2FF}'
        | '\u{370}'..='\u{37D}'
        | '\u{37F}'..='\u{1FFF}'
        | '\u{200C}'..='\u{200D}'
        | '\u{2070}'..='\u{218F}'
        | '\u{2C00}'..='\u{2FEF}'
        | '\u{3001}'..='\u{D7FF}'
        | '\u{F900}'..='\u{FDCF}'
        | '\u{FDF0}'..='\u{FFFD}'
        | '\u{10000}'..='\u{EFFFF}')
}

/// Returns `true` if `c` matches the XML `NameChar` production `[4a]`:
///
/// ```text
/// NameChar ::= NameStartChar | "-" | "." | [0-9] | #xB7
///            | [#x0300-#x036F] | [#x203F-#x2040]
/// ```
///
/// This production is identical in XML 1.0 (Fifth Edition) and XML 1.1.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::xml::is_name_char;
///
/// assert!(is_name_char('a'));
/// assert!(is_name_char('-'));
/// assert!(is_name_char('.'));
/// assert!(is_name_char('0'));
/// assert!(is_name_char('\u{B7}')); // middle dot
/// assert!(!is_name_char(' '));
/// ```
#[must_use]
pub fn is_name_char(c: char) -> bool {
    is_name_start_char(c)
        || matches!(c,
            '-' | '.' | '0'..='9' | '\u{B7}'
            | '\u{300}'..='\u{36F}'
            | '\u{203F}'..='\u{2040}')
}

/// Returns `true` if `c` is XML white space per the `S` production `[3]`:
///
/// ```text
/// S ::= (#x20 | #x9 | #xD | #xA)+
/// ```
///
/// The predicate matches a single white-space character; callers combine it to
/// recognise the one-or-more repetition of the production.
///
/// # Examples
///
/// ```
/// use oxixml_unicode::xml::is_xml_whitespace;
///
/// assert!(is_xml_whitespace(' '));
/// assert!(is_xml_whitespace('\t'));
/// assert!(is_xml_whitespace('\r'));
/// assert!(is_xml_whitespace('\n'));
/// assert!(!is_xml_whitespace('\u{A0}')); // no-break space is not XML S
/// ```
#[must_use]
pub fn is_xml_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}
