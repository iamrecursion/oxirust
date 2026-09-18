//! Boolean character-class predicates from the `Name`, `NCName`, `Nmtoken`, and `QName`
//! productions of [XML 1.0](https://www.w3.org/TR/xml/) and
//! [Namespaces in XML 1.0/1.1](https://www.w3.org/TR/xml-names/).
//!
//! These are pure grammar checks: they say nothing about whether a name is *bound* to a
//! namespace, or whether it is a reserved name like `xml` or `xmlns` — see [`QName`](crate::QName)
//! and [`NamespaceStack`](crate::NamespaceStack) for that.

use oxixml_unicode::xml::{is_name_char, is_name_start_char};

/// Returns `true` if `s` matches the XML `Name` production `[5]`:
///
/// ```text
/// Name ::= NameStartChar (NameChar)*
/// ```
///
/// A `Name` is the unqualified, namespace-unaware XML name grammar: it allows any number of
/// colons anywhere (a colon is itself a valid `NameStartChar`), unlike [`is_ncname`] or
/// [`is_qname`], which apply the additional Namespaces-in-XML restrictions.
///
/// # Examples
///
/// ```
/// use oxixml_qname::is_name;
///
/// assert!(is_name("book"));
/// assert!(is_name("book:title")); // colons are unrestricted at the `Name` level
/// assert!(is_name("_private"));
/// assert!(is_name("中文")); // CJK Unified Ideographs are NameStartChar
/// assert!(!is_name("")); // empty
/// assert!(!is_name("1st")); // a digit cannot start a Name
/// assert!(!is_name("a b")); // no whitespace
/// ```
#[must_use]
pub fn is_name(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if is_name_start_char(c) => chars.all(is_name_char),
        _ => false,
    }
}

/// Returns `true` if `s` matches the XML `Nmtoken` production `[7]`:
///
/// ```text
/// Nmtoken ::= (NameChar)+
/// ```
///
/// Unlike [`is_name`], an `Nmtoken` has no start-character restriction, so it may begin with a
/// digit, a combining mark, `-`, or `.`.
///
/// # Examples
///
/// ```
/// use oxixml_qname::is_nmtoken;
///
/// assert!(is_nmtoken("book"));
/// assert!(is_nmtoken("1.0")); // digits and '.' are fine anywhere in an Nmtoken
/// assert!(is_nmtoken("-42"));
/// assert!(!is_nmtoken("")); // empty
/// assert!(!is_nmtoken("a b")); // no whitespace
/// ```
#[must_use]
pub fn is_nmtoken(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_name_char)
}

/// A `NameStartChar` that is also legal as the first character of an `NCName`: everything except
/// the colon, which [Namespaces in XML §3](https://www.w3.org/TR/xml-names/#ns-decl) reserves as
/// the `QName` prefix separator.
fn is_ncname_start_char(c: char) -> bool {
    c != ':' && is_name_start_char(c)
}

/// A `NameChar` that is also legal inside an `NCName` (see [`is_ncname_start_char`]).
fn is_ncname_char(c: char) -> bool {
    c != ':' && is_name_char(c)
}

/// Returns `true` if `s` matches the [Namespaces in XML](https://www.w3.org/TR/xml-names/#NT-NCName)
/// `NCName` production:
///
/// ```text
/// NCName ::= Name - (Char* ':' Char*)   /* An XML Name, minus the ":" */
/// ```
///
/// equivalently spelled out as `NCNameStartChar (NCNameChar)*` where `NCNameStartChar` and
/// `NCNameChar` are `NameStartChar`/`NameChar` with the colon removed. An `NCName` is exactly
/// what may stand alone as a `QName` prefix or local part.
///
/// # Examples
///
/// ```
/// use oxixml_qname::is_ncname;
///
/// assert!(is_ncname("book"));
/// assert!(is_ncname("日本語")); // 5th-edition CJK NameStartChar range
/// assert!(is_ncname("a\u{10000}")); // astral-plane NameStartChar (U+10000..=U+EFFFF)
/// assert!(is_ncname("a\u{0301}")); // combining acute accent is a valid (non-start) NameChar
/// assert!(!is_ncname("\u{0301}a")); // ...but not as the first character
/// assert!(!is_ncname("book:title")); // colons are never allowed inside an NCName
/// assert!(!is_ncname("1st")); // a digit cannot start a Name
/// assert!(!is_ncname("")); // empty
/// ```
#[must_use]
pub fn is_ncname(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if is_ncname_start_char(c) => chars.all(is_ncname_char),
        _ => false,
    }
}

/// Splits `s` on its first colon, without validating either half.
///
/// Returns `None` for the empty string (never a valid `Name`, `NCName`, or `QName`). Otherwise
/// returns `Some((prefix, local))`, where `prefix` is `None` when `s` contains no colon.
///
/// Splitting on only the *first* colon is sufficient to detect a malformed `QName` with more
/// than one colon: [`is_ncname`] (and [`crate::QName::parse`]) reject any string containing a
/// colon, so a stray second colon that ends up inside `local` makes the corresponding `NCName`
/// check fail on its own — no separate "too many colons" check is needed.
pub(crate) fn split_qname(s: &str) -> Option<(Option<&str>, &str)> {
    if s.is_empty() {
        return None;
    }
    Some(match s.find(':') {
        Some(idx) => (Some(&s[..idx]), &s[idx + 1..]),
        None => (None, s),
    })
}

/// Returns `true` if `s` matches the [Namespaces in XML](https://www.w3.org/TR/xml-names/#NT-QName)
/// `QName` production:
///
/// ```text
/// QName         ::= PrefixedName | UnprefixedName
/// PrefixedName   ::= Prefix ':' LocalPart
/// UnprefixedName ::= LocalPart
/// Prefix         ::= NCName
/// LocalPart      ::= NCName
/// ```
///
/// A `QName` is a *lexical* qualified name: exactly zero or one colon, with both surrounding
/// parts (when present) valid `NCName`s. This says nothing about whether the prefix is actually
/// bound to a namespace — see [`crate::NamespaceStack::resolve_qname`] for that.
///
/// # Examples
///
/// ```
/// use oxixml_qname::is_qname;
///
/// assert!(is_qname("title")); // UnprefixedName
/// assert!(is_qname("book:title")); // PrefixedName
/// assert!(!is_qname("book:title:extra")); // more than one colon
/// assert!(!is_qname(":title")); // empty prefix
/// assert!(!is_qname("book:")); // empty local part
/// assert!(!is_qname("")); // empty
/// ```
#[must_use]
pub fn is_qname(s: &str) -> bool {
    match split_qname(s) {
        Some((Some(prefix), local)) => is_ncname(prefix) && is_ncname(local),
        Some((None, local)) => is_ncname(local),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_accepts_colons_anywhere() {
        assert!(is_name("a:b:c"));
        assert!(is_name(":leading"));
        assert!(is_name("trailing:"));
    }

    #[test]
    fn name_rejects_empty_and_bad_start() {
        assert!(!is_name(""));
        assert!(!is_name("0abc"));
        assert!(!is_name("-abc"));
        assert!(!is_name(".abc"));
        assert!(!is_name("\u{0301}abc")); // combining mark cannot start a Name
    }

    #[test]
    fn name_accepts_digits_and_combining_marks_mid_name() {
        assert!(is_name("a0"));
        assert!(is_name("a-b.c9"));
        assert!(is_name("a\u{0301}"));
        assert!(is_name("a\u{00B7}b")); // middle dot, #xB7
    }

    #[test]
    fn nmtoken_allows_any_name_char_first() {
        assert!(is_nmtoken("0.42"));
        assert!(is_nmtoken("-1"));
        assert!(is_nmtoken("\u{0301}"));
        assert!(!is_nmtoken(""));
        assert!(!is_nmtoken("a b")); // whitespace is never a NameChar
    }

    #[test]
    fn nmtoken_permits_colon_as_a_name_char() {
        // Nmtoken is built from NameChar, and ':' is a NameChar (unlike NCNameChar).
        assert!(is_nmtoken("a:b"));
    }

    #[test]
    fn ncname_rejects_any_colon() {
        assert!(!is_ncname("a:b"));
        assert!(!is_ncname(":a"));
        assert!(!is_ncname("a:"));
        assert!(!is_ncname(":"));
    }

    #[test]
    fn ncname_rejects_digit_first_char() {
        assert!(!is_ncname("0abc"));
        assert!(!is_ncname("9"));
    }

    #[test]
    fn ncname_accepts_cjk_and_astral_name_chars() {
        assert!(is_ncname("日本語"));
        assert!(is_ncname("中文标题"));
        // U+10000 is the first astral-plane NameStartChar (5th-edition range extension).
        assert!(is_ncname("\u{10000}"));
        assert!(is_ncname("a\u{10000}b"));
    }

    #[test]
    fn ncname_accepts_combining_chars_only_after_the_first_position() {
        assert!(is_ncname("a\u{0301}"));
        assert!(!is_ncname("\u{0301}a"));
    }

    #[test]
    fn ncname_rejects_empty() {
        assert!(!is_ncname(""));
    }

    #[test]
    fn qname_accepts_prefixed_and_unprefixed() {
        assert!(is_qname("title"));
        assert!(is_qname("book:title"));
        assert!(is_qname("日本語:タイトル"));
    }

    #[test]
    fn qname_rejects_malformed_colon_placement() {
        assert!(!is_qname(""));
        assert!(!is_qname(":title"));
        assert!(!is_qname("book:"));
        assert!(!is_qname("book:title:extra"));
        assert!(!is_qname(":"));
    }

    #[test]
    fn qname_rejects_invalid_ncname_parts() {
        assert!(!is_qname("0book:title"));
        assert!(!is_qname("book:0title"));
    }

    #[test]
    fn split_qname_handles_empty_and_unprefixed() {
        assert_eq!(split_qname(""), None);
        assert_eq!(split_qname("local"), Some((None, "local")));
        assert_eq!(split_qname("p:local"), Some((Some("p"), "local")));
        assert_eq!(split_qname("p:a:b"), Some((Some("p"), "a:b")));
    }
}
