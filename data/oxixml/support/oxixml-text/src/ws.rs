//! XML Schema whitespace facets ([XSD 1.1 Part 2 §4.3.6]).
//!
//! Three `whiteSpace` facet values are defined by XSD: `preserve` (no transformation — not
//! represented here, since it is simply the identity function), `replace`
//! ([`replace_ws`]), and `collapse` ([`collapse`]). Both operate over the four XML
//! whitespace characters recognized by [`is_xml_ws`]: `#x20` (space), `#x9` (tab), `#xD`
//! (carriage return), and `#xA` (line feed) — note this is **not** the full set of Unicode
//! whitespace, just the XML production `S`'s character class.
//!
//! [XSD 1.1 Part 2 §4.3.6]: https://www.w3.org/TR/xmlschema11-2/#rf-whiteSpace

use std::borrow::Cow;

/// Returns whether `c` is one of the four XML whitespace characters: `#x20`, `#x9`, `#xD`,
/// or `#xA`.
///
/// This is the character class matched by the XML `S` production, which is narrower than
/// [`char::is_whitespace`] (it excludes, for example, `#x85` NEL and all other Unicode
/// whitespace).
#[must_use]
pub const fn is_xml_ws(c: char) -> bool {
    matches!(c, '\u{20}' | '\u{9}' | '\u{D}' | '\u{A}')
}

/// Applies the XSD `replace` whitespace facet: every `#x9`, `#xD`, and `#xA` character is
/// replaced with `#x20`; the string's length in characters never changes.
///
/// Returns a borrowed [`Cow`] when `input` contains no character that needs replacing.
#[must_use]
pub fn replace_ws(input: &str) -> Cow<'_, str> {
    if !input.chars().any(|c| is_xml_ws(c) && c != ' ') {
        return Cow::Borrowed(input);
    }
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        out.push(if is_xml_ws(c) { ' ' } else { c });
    }
    Cow::Owned(out)
}

/// Applies the XSD `collapse` whitespace facet: equivalent to [`replace_ws`] followed by
/// trimming leading/trailing spaces and squeezing every internal run of one-or-more
/// whitespace characters down to a single `#x20`.
///
/// Returns a borrowed [`Cow`] when `input` is already in collapsed form.
#[must_use]
pub fn collapse(input: &str) -> Cow<'_, str> {
    let trimmed = input.trim_matches(is_xml_ws);

    let mut needs_alloc = false;
    let mut previous_was_ws = false;
    for c in trimmed.chars() {
        if is_xml_ws(c) {
            if c != ' ' || previous_was_ws {
                needs_alloc = true;
                break;
            }
            previous_was_ws = true;
        } else {
            previous_was_ws = false;
        }
    }

    if !needs_alloc {
        return Cow::Borrowed(trimmed);
    }

    let mut out = String::with_capacity(trimmed.len());
    let mut previous_was_ws = false;
    for c in trimmed.chars() {
        if is_xml_ws(c) {
            if !previous_was_ws {
                out.push(' ');
            }
            previous_was_ws = true;
        } else {
            out.push(c);
            previous_was_ws = false;
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_xml_ws_matches_exactly_four_characters() {
        for c in ['\u{20}', '\u{9}', '\u{D}', '\u{A}'] {
            assert!(is_xml_ws(c));
        }
        // Unicode whitespace outside the XML `S` production is excluded.
        assert!(!is_xml_ws('\u{A0}')); // NBSP
        assert!(!is_xml_ws('\u{85}')); // NEL
        assert!(!is_xml_ws('\u{2003}')); // EM SPACE
        assert!(!is_xml_ws('x'));
    }

    #[test]
    fn replace_turns_tab_cr_lf_into_space() {
        assert_eq!(replace_ws("a\tb\rc\nd"), "a b c d");
        assert_eq!(
            replace_ws("no whitespace change needed"),
            Cow::Borrowed("no whitespace change needed")
        );
        // Existing spaces and non-XML whitespace are left untouched.
        assert_eq!(replace_ws("a  b\u{A0}c"), "a  b\u{A0}c");
    }

    #[test]
    fn replace_preserves_length_in_characters() {
        let input = "a\t\tb\r\n\nc";
        let replaced = replace_ws(input);
        assert_eq!(replaced.chars().count(), input.chars().count());
    }

    #[test]
    fn collapse_trims_and_squeezes() {
        assert_eq!(collapse("  a\t b  \n c  "), "a b c");
        assert_eq!(collapse("already collapsed"), "already collapsed");
        assert_eq!(collapse(""), "");
        assert_eq!(collapse("   "), "");
        assert_eq!(collapse("single"), "single");
    }

    #[test]
    fn collapse_returns_borrowed_when_already_collapsed() {
        let input = "already collapsed text";
        assert!(matches!(collapse(input), Cow::Borrowed(_)));
    }

    #[test]
    fn collapse_returns_owned_when_changed() {
        let input = "needs  collapsing";
        assert!(matches!(collapse(input), Cow::Owned(_)));
    }

    #[test]
    fn collapse_mixed_whitespace_runs() {
        assert_eq!(collapse("a\t\r\n b"), "a b");
        assert_eq!(collapse("\t\ta\t\t"), "a");
    }

    #[test]
    fn collapse_preserves_non_xml_whitespace() {
        // NBSP is not XML whitespace, so it is neither trimmed nor squeezed.
        assert_eq!(collapse(" a\u{A0}b "), "a\u{A0}b");
    }
}
