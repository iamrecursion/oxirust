//! URI handling required by RELAX NG: the [XLink §5.4][xlink] escaping applied to
//! `href` and `datatypeLibrary` values, and a strict [RFC 2396][rfc2396]
//! `URI-reference` recognizer (as amended by [RFC 2732][rfc2732] for literal IPv6
//! addresses).
//!
//! RELAX NG deliberately talks about *URI references*, not IRIs: §3 says that a
//! value matching the `anyURI` symbol is "a string that, after escaping of
//! disallowed values as described in Section 5.4 of \[XLink], is a URI reference
//! as defined in \[RFC 2396]". A conforming implementation therefore has to
//! reject `http://example.com/%xx` (a malformed escape) while accepting
//! `http://example.com/%Aa`, and it has to know that `xyzzy/foo:bar` is
//! *relative* even though it contains a colon.
//!
//! [xlink]: https://www.w3.org/TR/xlink/#link-locators
//! [rfc2396]: https://www.rfc-editor.org/rfc/rfc2396
//! [rfc2732]: https://www.rfc-editor.org/rfc/rfc2732

/// Applies the XLink §5.4 escaping to `value`.
///
/// Every character that RFC 2396 excludes — all non-ASCII characters, the
/// control characters, the space, and ``<>"{}|\^` `` — is replaced by the
/// percent-encoding of its UTF-8 bytes. The number sign, the percent sign and the
/// square brackets are *not* escaped, exactly as XLink prescribes; this is what
/// makes a stray `%` an error rather than a literal.
#[must_use]
pub(crate) fn xlink_escape(value: &str) -> String {
    fn needs_escape(c: char) -> bool {
        if !c.is_ascii() {
            return true;
        }
        matches!(
            c,
            '\u{0}'..='\u{20}' | '\u{7f}' | '<' | '>' | '"' | '{' | '}' | '|' | '\\' | '^' | '`'
        )
    }

    if !value.chars().any(needs_escape) {
        return value.to_owned();
    }
    let mut out = String::with_capacity(value.len() + 8);
    let mut buf = [0_u8; 4];
    for c in value.chars() {
        if needs_escape(c) {
            for byte in c.encode_utf8(&mut buf).as_bytes() {
                out.push('%');
                out.push(hex_digit(byte >> 4));
                out.push(hex_digit(byte & 0x0f));
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn hex_digit(nibble: u8) -> char {
    char::from(match nibble {
        0..=9 => b'0' + nibble,
        _ => b'A' + (nibble - 10),
    })
}

/// The structural classification of a string that is a valid `URI-reference`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UriReferenceInfo {
    /// `true` when the reference starts with a `scheme ":"`, i.e. is absolute.
    pub absolute: bool,
    /// `true` when the reference contains a `#`, i.e. carries a fragment
    /// identifier (even an empty one).
    pub has_fragment: bool,
}

/// Recognizes an RFC 2396 `URI-reference`, returning how it is structured.
///
/// Returns `None` when `value` is not a `URI-reference` at all.
#[must_use]
pub(crate) fn parse_uri_reference(value: &str) -> Option<UriReferenceInfo> {
    let (before_fragment, fragment) = match value.find('#') {
        Some(index) => (&value[..index], Some(&value[index + 1..])),
        None => (value, None),
    };
    if let Some(fragment) = fragment
        && !fragment.chars().all(is_uric)
    {
        return None;
    }

    if before_fragment.is_empty() {
        // `URI-reference = [ absoluteURI | relativeURI ] [ "#" fragment ]`: both
        // halves are optional, so the empty string is a (relative) reference.
        return Some(UriReferenceInfo {
            absolute: false,
            has_fragment: fragment.is_some(),
        });
    }

    let absolute = match scheme_length(before_fragment) {
        Some(length) => {
            if !is_absolute_uri_tail(&before_fragment[length + 1..]) {
                return None;
            }
            true
        }
        None => {
            if !is_relative_uri(before_fragment) {
                return None;
            }
            false
        }
    };

    Some(UriReferenceInfo {
        absolute,
        has_fragment: fragment.is_some(),
    })
}

/// Returns the length of the `scheme` production at the start of `value`, when
/// `value` really does start with `scheme ":"`.
fn scheme_length(value: &str) -> Option<usize> {
    let bytes = value.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_alphabetic() {
        return None;
    }
    let mut index = 1;
    while index < bytes.len() {
        match bytes[index] {
            b':' => return Some(index),
            b if b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.') => index += 1,
            _ => return None,
        }
    }
    None
}

/// `hier_part | opaque_part`, i.e. everything after `scheme ":"`.
fn is_absolute_uri_tail(tail: &str) -> bool {
    if tail.starts_with('/') {
        return is_hier_part(tail);
    }
    // opaque_part = uric_no_slash *uric
    let Some(first) = tail.chars().next() else {
        return false;
    };
    if !is_uric_no_slash(first) && first != '%' {
        return false;
    }
    all_of(tail, is_uric)
}

/// `( net_path | abs_path ) [ "?" query ]`
fn is_hier_part(value: &str) -> bool {
    let (path, query) = split_query(value);
    if let Some(query) = query
        && !all_uric(query)
    {
        return false;
    }
    if let Some(rest) = path.strip_prefix("//") {
        // net_path = "//" authority [ abs_path ]
        let (authority, abs_path) = match rest.find('/') {
            Some(index) => (&rest[..index], Some(&rest[index..])),
            None => (rest, None),
        };
        if !is_authority(authority) {
            return false;
        }
        match abs_path {
            Some(abs_path) => is_abs_path(abs_path),
            None => true,
        }
    } else {
        is_abs_path(path)
    }
}

/// `( net_path | abs_path | rel_path ) [ "?" query ]`
fn is_relative_uri(value: &str) -> bool {
    let (path, query) = split_query(value);
    if let Some(query) = query
        && !all_uric(query)
    {
        return false;
    }
    if path.starts_with("//") || path.starts_with('/') {
        return is_hier_part(path);
    }
    // rel_path = rel_segment [ abs_path ]
    let (segment, abs_path) = match path.find('/') {
        Some(index) => (&path[..index], Some(&path[index..])),
        None => (path, None),
    };
    if segment.is_empty() || !all_of(segment, is_rel_segment_char) {
        return false;
    }
    match abs_path {
        Some(abs_path) => is_abs_path(abs_path),
        None => true,
    }
}

fn split_query(value: &str) -> (&str, Option<&str>) {
    match value.find('?') {
        Some(index) => (&value[..index], Some(&value[index + 1..])),
        None => (value, None),
    }
}

/// `abs_path = "/" path_segments`; `segment` characters are `pchar`, `;` and `:`.
fn is_abs_path(value: &str) -> bool {
    match value.strip_prefix('/') {
        Some(rest) => rest
            .split('/')
            .all(|segment| all_of(segment, is_segment_char)),
        None => false,
    }
}

/// `authority = server | reg_name`. Both are recognized by their character sets;
/// RFC 2732 additionally allows a bracketed IPv6 literal in the host position.
fn is_authority(value: &str) -> bool {
    if value.is_empty() {
        // server can be empty (e.g. "file:///path").
        return true;
    }
    let (userinfo, hostport) = match value.rfind('@') {
        Some(index) => (Some(&value[..index]), &value[index + 1..]),
        None => (None, value),
    };
    if let Some(userinfo) = userinfo
        && !all_of(userinfo, is_userinfo_char)
    {
        return all_of(value, is_reg_name_char);
    }
    if is_hostport(hostport) {
        return true;
    }
    all_of(value, is_reg_name_char)
}

fn is_hostport(value: &str) -> bool {
    let (host, port) = if let Some(rest) = value.strip_prefix('[') {
        // RFC 2732 IPv6reference.
        match rest.find(']') {
            Some(index) => {
                let host = &rest[..index];
                if host.is_empty()
                    || !host
                        .chars()
                        .all(|c| c.is_ascii_hexdigit() || c == ':' || c == '.')
                {
                    return false;
                }
                let tail = &rest[index + 1..];
                match tail.strip_prefix(':') {
                    Some(port) => (host, Some(port)),
                    None if tail.is_empty() => (host, None),
                    None => return false,
                }
            }
            None => return false,
        }
    } else {
        match value.rfind(':') {
            Some(index) => (&value[..index], Some(&value[index + 1..])),
            None => (value, None),
        }
    };
    if let Some(port) = port
        && !port.chars().all(|c| c.is_ascii_digit())
    {
        return false;
    }
    is_host(host)
}

/// `host = hostname | IPv4address`, both recognized as dot-separated
/// alphanumeric-or-hyphen labels.
fn is_host(value: &str) -> bool {
    if value.is_empty() {
        return true;
    }
    let body = value.strip_suffix('.').unwrap_or(value);
    body.split('.').all(|label| {
        !label.is_empty()
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
    })
}

fn is_uric(c: char) -> bool {
    is_reserved(c) || is_unreserved(c)
}

fn is_reserved(c: char) -> bool {
    matches!(
        c,
        ';' | '/' | '?' | ':' | '@' | '&' | '=' | '+' | '$' | ',' | '[' | ']'
    )
}

fn is_unreserved(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '!' | '~' | '*' | '\'' | '(' | ')')
}

fn is_uric_no_slash(c: char) -> bool {
    is_unreserved(c) || matches!(c, ';' | '?' | ':' | '@' | '&' | '=' | '+' | '$' | ',')
}

fn is_pchar(c: char) -> bool {
    is_unreserved(c) || matches!(c, ':' | '@' | '&' | '=' | '+' | '$' | ',')
}

fn is_segment_char(c: char) -> bool {
    is_pchar(c) || c == ';'
}

fn is_rel_segment_char(c: char) -> bool {
    is_unreserved(c) || matches!(c, ';' | '@' | '&' | '=' | '+' | '$' | ',')
}

fn is_userinfo_char(c: char) -> bool {
    is_unreserved(c) || matches!(c, ';' | ':' | '&' | '=' | '+' | '$' | ',')
}

fn is_reg_name_char(c: char) -> bool {
    is_unreserved(c) || matches!(c, '$' | ',' | ';' | ':' | '@' | '&' | '=' | '+')
}

fn all_uric(value: &str) -> bool {
    all_of(value, is_uric)
}

/// Checks that every character of `value` satisfies `allowed`, treating `%HH`
/// escape triples as a single allowed unit and rejecting malformed escapes.
fn all_of(value: &str, allowed: fn(char) -> bool) -> bool {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len()
                || !bytes[index + 1].is_ascii_hexdigit()
                || !bytes[index + 2].is_ascii_hexdigit()
            {
                return false;
            }
            index += 3;
            continue;
        }
        let Some(c) = value[index..].chars().next() else {
            return false;
        };
        if !allowed(c) {
            return false;
        }
        index += c.len_utf8();
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(value: &str) -> Option<UriReferenceInfo> {
        parse_uri_reference(&xlink_escape(value))
    }

    #[test]
    fn absolute_references() {
        assert_eq!(
            info("http://www.example.com/foo"),
            Some(UriReferenceInfo {
                absolute: true,
                has_fragment: false
            })
        );
        assert_eq!(
            info("http:ok"),
            Some(UriReferenceInfo {
                absolute: true,
                has_fragment: false
            })
        );
        assert_eq!(
            info("foo:bar"),
            Some(UriReferenceInfo {
                absolute: true,
                has_fragment: false
            })
        );
        assert_eq!(
            info("foobar:xyzzy"),
            Some(UriReferenceInfo {
                absolute: true,
                has_fragment: false
            })
        );
    }

    #[test]
    fn relative_references() {
        assert!(matches!(info("xyzzy"), Some(i) if !i.absolute));
        assert!(matches!(info("xyzzy?foo:bar"), Some(i) if !i.absolute));
        assert!(matches!(info("xyzzy/foo:bar"), Some(i) if !i.absolute));
        assert!(matches!(info("xyzzy#foo:bar"), Some(i) if !i.absolute && i.has_fragment));
        // An underscore is not allowed in a scheme, and a `rel_segment` may not
        // contain a colon, so this is not a URI reference at all.
        assert_eq!(info("foo_bar:xyzzy"), None);
    }

    #[test]
    fn fragments_are_detected_even_when_empty() {
        assert!(matches!(info("http://www.example.com#"), Some(i) if i.has_fragment));
        assert!(matches!(info("http://www.example.com#xyzzy"), Some(i) if i.has_fragment));
    }

    #[test]
    fn malformed_escapes_are_rejected() {
        assert_eq!(info("http://www.example.com/%"), None);
        assert_eq!(info("http://www.example.com/%xx"), None);
        assert!(info("http://www.example.com/%Aa").is_some());
    }

    #[test]
    fn empty_opaque_part_is_not_a_uri() {
        assert_eq!(info("foo:"), None);
    }

    #[test]
    fn disallowed_characters_are_escaped_first() {
        assert_eq!(xlink_escape("a b"), "a%20b");
        assert_eq!(xlink_escape("caf\u{e9}"), "caf%C3%A9");
        assert_eq!(xlink_escape("a#b%c[d]"), "a#b%c[d]");
        assert!(info("http://www.example.com/a b").is_some());
    }

    #[test]
    fn empty_string_is_a_relative_reference() {
        assert!(matches!(info(""), Some(i) if !i.absolute && !i.has_fragment));
    }
}
