//! Character-level productions of the XML specification.
//!
//! The predicates in this module implement the grammar of [XML 1.0 (5th edition)]
//! and [XML 1.1] verbatim. They are used by the optional well-formedness checks
//! of the [`Reader`](crate::reader::Reader) and never by the default,
//! `quick-xml`-compatible parsing path.
//!
//! [XML 1.0 (5th edition)]: https://www.w3.org/TR/xml/
//! [XML 1.1]: https://www.w3.org/TR/xml11/

use crate::XmlVersion;

/// Implements the XML 1.0 [`Char`] production.
///
/// ```text
/// Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
/// ```
///
/// [`Char`]: https://www.w3.org/TR/xml/#NT-Char
#[inline]
pub(crate) const fn is_xml10_char(ch: char) -> bool {
    matches!(ch,
        '\u{09}' | '\u{0A}' | '\u{0D}'
        | '\u{20}'..='\u{D7FF}'
        | '\u{E000}'..='\u{FFFD}'
        | '\u{10000}'..='\u{10FFFF}')
}

/// Implements the XML 1.1 [`Char`] production.
///
/// ```text
/// Char ::= [#x1-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
/// ```
///
/// [`Char`]: https://www.w3.org/TR/xml11/#NT-Char
#[inline]
pub(crate) const fn is_xml11_char(ch: char) -> bool {
    matches!(ch,
        '\u{01}'..='\u{D7FF}'
        | '\u{E000}'..='\u{FFFD}'
        | '\u{10000}'..='\u{10FFFF}')
}

/// Implements the XML 1.1 [`RestrictedChar`] production.
///
/// ```text
/// RestrictedChar ::= [#x1-#x8] | [#xB-#xC] | [#xE-#x1F] | [#x7F-#x84] | [#x86-#x9F]
/// ```
///
/// Those characters match the XML 1.1 `Char` production — so they may be written
/// as character references — but [section 2.1] subtracts them from the `document`
/// production, which makes their *literal* presence a fatal error:
///
/// ```text
/// document ::= ( prolog element Misc* ) - ( Char* RestrictedChar Char* )
/// ```
///
/// [`RestrictedChar`]: https://www.w3.org/TR/xml11/#NT-RestrictedChar
/// [section 2.1]: https://www.w3.org/TR/xml11/#sec-well-formed
#[inline]
pub(crate) const fn is_xml11_restricted_char(ch: char) -> bool {
    matches!(ch,
        '\u{01}'..='\u{08}'
        | '\u{0B}'..='\u{0C}'
        | '\u{0E}'..='\u{1F}'
        | '\u{7F}'..='\u{84}'
        | '\u{86}'..='\u{9F}')
}

/// Dispatches [`is_xml10_char`] / [`is_xml11_char`] on the document's version.
///
/// This is the predicate for characters that arrive through a *character
/// reference*: XML 1.1 `RestrictedChar`s are legal that way. Use
/// [`is_literal_char`] for characters that appear literally in the document.
#[inline]
pub(crate) const fn is_char(ch: char, version: XmlVersion) -> bool {
    match version {
        XmlVersion::Explicit1_1 => is_xml11_char(ch),
        _ => is_xml10_char(ch),
    }
}

/// Whether `ch` may appear *literally* in a document of the given version.
///
/// This is [`is_char`] minus the XML 1.1 [`RestrictedChar`] production, as
/// required by the `document` production of XML 1.1 [section 2.1].
///
/// [`RestrictedChar`]: https://www.w3.org/TR/xml11/#NT-RestrictedChar
/// [section 2.1]: https://www.w3.org/TR/xml11/#sec-well-formed
#[inline]
pub(crate) const fn is_literal_char(ch: char, version: XmlVersion) -> bool {
    match version {
        XmlVersion::Explicit1_1 => is_xml11_char(ch) && !is_xml11_restricted_char(ch),
        _ => is_xml10_char(ch),
    }
}

/// Implements the [`S`] production: `#x20 | #x9 | #xD | #xA`.
///
/// [`S`]: https://www.w3.org/TR/xml/#NT-S
#[inline]
pub(crate) const fn is_space(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\r' | '\n')
}

/// Implements the [`NameStartChar`] production of XML 1.0 5th edition, which is
/// identical to the XML 1.1 one.
///
/// [`NameStartChar`]: https://www.w3.org/TR/xml/#NT-NameStartChar
#[inline]
pub(crate) const fn is_name_start_char(ch: char) -> bool {
    matches!(ch,
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

/// Implements the [`NameChar`] production of XML 1.0 5th edition.
///
/// [`NameChar`]: https://www.w3.org/TR/xml/#NT-NameChar
#[inline]
pub(crate) const fn is_name_char(ch: char) -> bool {
    is_name_start_char(ch)
        || matches!(ch,
            '-' | '.' | '0'..='9' | '\u{B7}'
            | '\u{0300}'..='\u{036F}'
            | '\u{203F}'..='\u{2040}')
}

/// Implements the [`Name`] production: `NameStartChar (NameChar)*`.
///
/// [`Name`]: https://www.w3.org/TR/xml/#NT-Name
pub(crate) fn is_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if is_name_start_char(first) => chars.all(is_name_char),
        _ => false,
    }
}

/// Implements the [`NCName`] production of Namespaces in XML: a `Name` that
/// contains no colon.
///
/// [`NCName`]: https://www.w3.org/TR/xml-names/#NT-NCName
pub(crate) fn is_ncname(name: &str) -> bool {
    is_name(name) && !name.contains(':')
}

/// Implements the [`QName`] production of Namespaces in XML:
/// `PrefixedName | UnprefixedName`, that is `NCName (':' NCName)?`.
///
/// [`QName`]: https://www.w3.org/TR/xml-names/#NT-QName
pub(crate) fn is_qname(name: &str) -> bool {
    match name.split_once(':') {
        Some((prefix, local)) => is_ncname(prefix) && is_ncname(local),
        None => is_ncname(name),
    }
}

/// Implements the [`Nmtoken`] production: `(NameChar)+`.
///
/// [`Nmtoken`]: https://www.w3.org/TR/xml/#NT-Nmtoken
pub(crate) fn is_nmtoken(token: &str) -> bool {
    !token.is_empty() && token.chars().all(is_name_char)
}

/// Implements the [`PubidChar`] production.
///
/// ```text
/// PubidChar ::= #x20 | #xD | #xA | [a-zA-Z0-9] | [-'()+,./:=?;!*#@$_%]
/// ```
///
/// [`PubidChar`]: https://www.w3.org/TR/xml/#NT-PubidChar
#[inline]
pub(crate) const fn is_pubid_char(ch: char) -> bool {
    matches!(ch,
        ' ' | '\r' | '\n'
        | 'a'..='z' | 'A'..='Z' | '0'..='9'
        | '-' | '\'' | '(' | ')' | '+' | ',' | '.' | '/' | ':'
        | '=' | '?' | ';' | '!' | '*' | '#' | '@' | '$' | '_' | '%')
}

/// Implements the [`EncName`] production: `[A-Za-z] ([A-Za-z0-9._] | '-')*`.
///
/// [`EncName`]: https://www.w3.org/TR/xml/#NT-EncName
pub(crate) fn is_enc_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() => {
            chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
        }
        _ => false,
    }
}

/// Implements the [`VersionNum`] production of XML 1.0: `'1.' [0-9]+`.
///
/// [`VersionNum`]: https://www.w3.org/TR/xml/#NT-VersionNum
pub(crate) fn is_version_num(version: &str) -> bool {
    match version.strip_prefix("1.") {
        Some(minor) => !minor.is_empty() && minor.bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_production() {
        assert!(!is_xml10_char('\u{0}'));
        assert!(!is_xml10_char('\u{C}'));
        assert!(is_xml10_char('\t'));
        assert!(is_xml10_char('\u{10FFFF}'));
        assert!(!is_xml11_char('\u{0}'));
        assert!(is_xml11_char('\u{C}'));
    }

    #[test]
    fn restricted_char_production() {
        // Every code point of the `RestrictedChar` production.
        for ch in [
            '\u{1}', '\u{8}', '\u{B}', '\u{C}', '\u{E}', '\u{1F}', '\u{7F}',
        ]
        .into_iter()
        .chain(['\u{84}', '\u{86}', '\u{9F}'])
        {
            assert!(is_xml11_restricted_char(ch), "U+{:04X}", ch as u32);
            // `RestrictedChar` is a subset of `Char`: legal via a character
            // reference, illegal literally.
            assert!(is_xml11_char(ch), "U+{:04X}", ch as u32);
            assert!(
                !is_literal_char(ch, XmlVersion::Explicit1_1),
                "U+{:04X}",
                ch as u32
            );
        }
        // Neighbours that are *not* restricted.
        for ch in ['\u{9}', '\u{A}', '\u{D}', '\u{20}', '\u{85}', '\u{A0}'] {
            assert!(!is_xml11_restricted_char(ch), "U+{:04X}", ch as u32);
            assert!(
                is_literal_char(ch, XmlVersion::Explicit1_1),
                "U+{:04X}",
                ch as u32
            );
        }
        // XML 1.0 is unaffected.
        assert!(is_literal_char('\u{85}', XmlVersion::Explicit1_0));
        assert!(!is_literal_char('\u{C}', XmlVersion::Explicit1_0));
    }

    #[test]
    fn name_production() {
        assert!(is_name("doc"));
        assert!(is_name(":doc"));
        assert!(is_name("a-b.c_d"));
        assert!(!is_name(""));
        assert!(!is_name(".doc"));
        assert!(!is_name("0A"));
        assert!(!is_name("A@"));
        assert!(!is_name("a b"));
    }

    #[test]
    fn namespace_productions() {
        assert!(is_ncname("foo"));
        assert!(!is_ncname("a:b"));
        assert!(is_qname("a:b"));
        assert!(!is_qname("a:b:c"));
        assert!(!is_qname("foo:"));
        assert!(!is_qname(":foo"));
        assert!(!is_qname("xmlns:"));
    }

    #[test]
    fn nmtoken_production() {
        assert!(is_nmtoken("0A"));
        assert!(!is_nmtoken(""));
        assert!(!is_nmtoken("a@"));
    }

    #[test]
    fn literals() {
        assert!(is_pubid_char('\''));
        assert!(!is_pubid_char('"'));
        assert!(!is_pubid_char('['));
        assert!(is_enc_name("UTF-8"));
        assert!(!is_enc_name(" UTF-8"));
        assert!(!is_enc_name("-8"));
        assert!(is_version_num("1.0"));
        assert!(is_version_num("1.10"));
        assert!(!is_version_num("1."));
        assert!(!is_version_num("1.0 "));
        assert!(!is_version_num("2.0"));
    }
}
