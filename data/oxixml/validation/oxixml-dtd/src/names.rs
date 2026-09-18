//! Character-class predicates from the XML 1.0 grammar.
//!
//! These are the productions the DTD scanner needs in order to know where a
//! token ends. Whole tokens are then validated with
//! [`oxixml_qname::is_name`] / [`oxixml_qname::is_nmtoken`], so the two agree by
//! construction.

/// The `S` production, XML 1.0 §2.3 `[3]`.
#[must_use]
pub(crate) const fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

/// The `NameStartChar` production, XML 1.0 §2.3 `[4]`.
#[must_use]
pub(crate) const fn is_name_start_char(c: char) -> bool {
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

/// The `NameChar` production, XML 1.0 §2.3 `[4a]`.
#[must_use]
pub(crate) const fn is_name_char(c: char) -> bool {
    is_name_start_char(c)
        || matches!(c,
            '-' | '.' | '0'..='9'
            | '\u{B7}'
            | '\u{300}'..='\u{36F}'
            | '\u{203F}'..='\u{2040}')
}

/// The `PubidChar` production, XML 1.0 §2.3 `[13]`.
#[must_use]
pub(crate) const fn is_pubid_char(c: char) -> bool {
    matches!(c,
        ' ' | '\r' | '\n'
        | 'a'..='z' | 'A'..='Z' | '0'..='9'
        | '-' | '\'' | '(' | ')' | '+' | ',' | '.' | '/' | ':' | '=' | '?'
        | ';' | '!' | '*' | '#' | '@' | '$' | '_' | '%')
}

/// Normalizes line ends to `\n`, as XML 1.0 §2.11 requires before parsing.
#[must_use]
pub(crate) fn normalize_line_ends(text: &str) -> String {
    if !text.contains('\r') {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            out.push('\n');
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_predicates_agree_with_oxixml_qname() {
        for name in ["a", "_x", ":q", "élan", "a-b.c0", "\u{4E2D}\u{6587}"] {
            assert!(oxixml_qname::is_name(name), "{name}");
            let mut chars = name.chars();
            let Some(first) = chars.next() else {
                continue;
            };
            assert!(is_name_start_char(first), "{name}");
            assert!(chars.all(is_name_char), "{name}");
        }
        assert!(!is_name_start_char('-'));
        assert!(!is_name_start_char('0'));
        assert!(is_name_char('-'));
        assert!(is_name_char('0'));
    }

    #[test]
    fn line_ends_normalize_to_line_feed() {
        assert_eq!(normalize_line_ends("a\r\nb\rc\nd"), "a\nb\nc\nd");
        assert_eq!(normalize_line_ends("plain"), "plain");
    }

    #[test]
    fn pubid_chars_exclude_double_quote_and_angle_brackets() {
        assert!(is_pubid_char('a'));
        assert!(is_pubid_char('/'));
        assert!(!is_pubid_char('"'));
        assert!(!is_pubid_char('<'));
        assert!(!is_pubid_char('\t'));
    }
}
