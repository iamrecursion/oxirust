//! Finding — and rewriting — namespace prefixes that hide inside content.
//!
//! Exclusive canonicalization only sees the prefixes that appear in *names*, so
//! a `QName` sitting in an attribute value (`xsi:type="xsd:string"`) or in an
//! XPath expression loses its declaration and with it its meaning. Canonical XML
//! 2.0 closes that hole with the `QNameAware` parameter, which names the
//! elements and attributes whose content is a `QName` or an XPath expression;
//! their prefixes then count as
//! [visibly utilized](https://www.w3.org/TR/xml-c14n2/#sec-Namespace-Processing)
//! and get rewritten along with everything else.
//!
//! The XPath scan is the one specified in
//! [§2.5.3](https://www.w3.org/TR/xml-c14n2/#sec-Namespace-Processing-Algorithm):
//! ignore single colons inside quoted strings, ignore the `::` of an axis
//! specifier, and for every remaining colon walk backwards over white space and
//! then over `NCName` characters to collect the prefix.

/// Splits a `QName` value into its optional prefix and its local part.
pub(crate) fn split_qname(value: &str) -> (Option<&str>, &str) {
    match value.split_once(':') {
        Some((prefix, local)) if !prefix.is_empty() && !local.is_empty() => (Some(prefix), local),
        _ => (None, value),
    }
}

/// Returns the prefix of a `QName`-valued piece of content, or `""` when it has
/// none (in which case it visibly utilizes the default namespace).
pub(crate) fn qname_prefix(value: &str) -> &str {
    let trimmed = value.trim();
    split_qname(trimmed).0.unwrap_or("")
}

/// Rewrites the prefix of a `QName`-valued piece of content.
///
/// `lookup` maps an original prefix (`""` for "no prefix", meaning the default
/// namespace) to its replacement; returning `None` leaves the value alone.
pub(crate) fn rewrite_qname<F>(value: &str, lookup: F) -> String
where
    F: Fn(&str) -> Option<String>,
{
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return value.to_owned();
    }
    let (prefix, local) = split_qname(trimmed);
    match lookup(prefix.unwrap_or("")) {
        Some(replacement) => {
            let leading = &value[..value.len() - value.trim_start().len()];
            let trailing = &value[value.trim_end().len()..];
            format!("{leading}{replacement}:{local}{trailing}")
        }
        None => value.to_owned(),
    }
}

/// The byte range of one prefix found in an XPath expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefixSpan {
    start: usize,
    end: usize,
}

/// Locates every namespace prefix used in an XPath 1.0 expression.
fn scan_xpath(expression: &str) -> Vec<PrefixSpan> {
    let chars: Vec<(usize, char)> = expression.char_indices().collect();
    let mut spans = Vec::new();
    let mut quote: Option<char> = None;
    let mut index = 0;

    while index < chars.len() {
        let (_, ch) = chars[index];
        if let Some(open) = quote {
            if ch == open {
                quote = None;
            }
            index += 1;
            continue;
        }
        if ch == '"' || ch == '\'' {
            quote = Some(ch);
            index += 1;
            continue;
        }
        if ch != ':' {
            index += 1;
            continue;
        }
        // `::` is an axis separator, never a prefix delimiter.
        if chars.get(index + 1).is_some_and(|(_, next)| *next == ':') {
            index += 2;
            continue;
        }

        let mut cursor = index;
        while cursor > 0 && chars[cursor - 1].1.is_whitespace() {
            cursor -= 1;
        }
        let end = cursor;
        while cursor > 0 && is_ncname_char(chars[cursor - 1].1) {
            cursor -= 1;
        }
        // An `NCName` may not start with a digit, a hyphen or a full stop.
        while cursor < end && !is_ncname_start_char(chars[cursor].1) {
            cursor += 1;
        }
        if cursor < end {
            let (start_byte, _) = chars[cursor];
            let (last_byte, last_char) = chars[end - 1];
            spans.push(PrefixSpan {
                start: start_byte,
                end: last_byte + last_char.len_utf8(),
            });
        }
        index += 1;
    }
    spans
}

/// Returns every namespace prefix visibly utilized by an XPath expression, in
/// order of first appearance and without duplicates.
pub(crate) fn xpath_prefixes(expression: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for span in scan_xpath(expression) {
        let prefix = &expression[span.start..span.end];
        if !out.iter().any(|seen| seen == prefix) {
            out.push(prefix.to_owned());
        }
    }
    out
}

/// Replaces every namespace prefix in an XPath expression, leaving quoted
/// strings, axis specifiers and everything else byte-for-byte intact.
pub(crate) fn rewrite_xpath_prefixes<F>(expression: &str, lookup: F) -> String
where
    F: Fn(&str) -> Option<String>,
{
    let mut out = String::with_capacity(expression.len());
    let mut copied = 0;
    for span in scan_xpath(expression) {
        let prefix = &expression[span.start..span.end];
        if let Some(replacement) = lookup(prefix) {
            out.push_str(&expression[copied..span.start]);
            out.push_str(&replacement);
            copied = span.end;
        }
    }
    out.push_str(&expression[copied..]);
    out
}

/// The `NameStartChar` production of
/// [XML 1.0 Fifth Edition §2.3](https://www.w3.org/TR/xml/#NT-NameStartChar),
/// minus the colon — that is, `NCNameStartChar`.
const fn is_ncname_start_char(ch: char) -> bool {
    matches!(ch,
        'A'..='Z'
        | '_'
        | 'a'..='z'
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

/// The `NameChar` production, minus the colon — that is, `NCNameChar`.
const fn is_ncname_char(ch: char) -> bool {
    is_ncname_start_char(ch)
        || matches!(ch,
            '-' | '.'
            | '0'..='9'
            | '\u{B7}'
            | '\u{300}'..='\u{36F}'
            | '\u{203F}'..='\u{2040}')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The expression from the W3C Canonical XML 2.0 test case `inNsContent`.
    const EXPRESSION: &str =
        "/soap-env:body/child::b:foo[@att1 != \"c:val\" and @att2 != 'xsd:string']";

    #[test]
    fn quoted_strings_and_axes_are_not_prefixes() {
        assert_eq!(xpath_prefixes(EXPRESSION), ["soap-env", "b"]);
    }

    #[test]
    fn rewriting_leaves_quoted_strings_alone() {
        let rewritten = rewrite_xpath_prefixes(EXPRESSION, |prefix| match prefix {
            "soap-env" => Some("n3".to_owned()),
            "b" => Some("n2".to_owned()),
            _ => None,
        });
        assert_eq!(
            rewritten,
            "/n3:body/child::n2:foo[@att1 != \"c:val\" and @att2 != 'xsd:string']"
        );
    }

    #[test]
    fn white_space_before_the_colon_is_skipped() {
        assert_eq!(xpath_prefixes("/soap  :  Body"), ["soap"]);
    }

    #[test]
    fn a_bare_colon_has_no_prefix() {
        assert!(xpath_prefixes("/:name").is_empty());
        assert!(xpath_prefixes("self::node()").is_empty());
    }

    #[test]
    fn qnames_split_on_the_first_colon() {
        assert_eq!(split_qname("xsd:string"), (Some("xsd"), "string"));
        assert_eq!(split_qname("string"), (None, "string"));
        assert_eq!(qname_prefix("  xsd:string  "), "xsd");
        assert_eq!(qname_prefix("string"), "");
    }

    #[test]
    fn qname_rewriting_preserves_surrounding_space() {
        let rewritten = rewrite_qname(" xsd:string ", |prefix| {
            (prefix == "xsd").then(|| "n1".to_owned())
        });
        assert_eq!(rewritten, " n1:string ");
    }
}
