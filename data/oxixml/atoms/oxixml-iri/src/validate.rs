//! Character-class predicates and per-component validation for the RFC 3987 IRI grammar.
//!
//! Each `validate_*` function walks a single component (scheme, authority, path, query,
//! fragment) once, byte by byte, accepting ASCII bytes through a precomputed
//! [`oxixml_text::percent::AsciiSet`] membership test, percent-encoded triplets through a
//! three-byte lookahead, and non-ASCII bytes through the relevant RFC 3987 `ucschar` /
//! `iprivate` ranges (decoded as a whole [`char`] so multi-byte UTF-8 sequences are classified
//! once rather than byte by byte).

use std::net::Ipv6Addr;
use std::str::FromStr;

use oxixml_support::bytes::memchr;
use oxixml_text::percent::AsciiSet;

use std::ops::Deref;

use crate::error::{IriParseError, IriParseErrorKind};
use crate::iri::Iri;
use crate::iri_ref::IriRef;

/// [RFC 3987 §2.2] `ucschar`: the non-ASCII code points that may appear directly (unescaped)
/// in an IRI, outside of the query component (which additionally allows `iprivate`, see
/// [`is_iprivate`]).
///
/// ```text
/// ucschar = %xA0-D7FF / %xF900-FDCF / %xFDF0-FFEF
///         / %x10000-1FFFD / %x20000-2FFFD / %x30000-3FFFD
///         / %x40000-4FFFD / %x50000-5FFFD / %x60000-6FFFD
///         / %x70000-7FFFD / %x80000-8FFFD / %x90000-9FFFD
///         / %xA0000-AFFFD / %xB0000-BFFFD / %xC0000-CFFFD
///         / %xD0000-DFFFD / %xE1000-EFFFD
/// ```
///
/// [RFC 3987 §2.2]: https://www.rfc-editor.org/rfc/rfc3987#section-2.2
#[inline]
#[must_use]
pub(crate) fn is_ucschar(c: char) -> bool {
    matches!(
        c,
        '\u{A0}'..='\u{D7FF}'
        | '\u{F900}'..='\u{FDCF}'
        | '\u{FDF0}'..='\u{FFEF}'
        | '\u{10000}'..='\u{1FFFD}'
        | '\u{20000}'..='\u{2FFFD}'
        | '\u{30000}'..='\u{3FFFD}'
        | '\u{40000}'..='\u{4FFFD}'
        | '\u{50000}'..='\u{5FFFD}'
        | '\u{60000}'..='\u{6FFFD}'
        | '\u{70000}'..='\u{7FFFD}'
        | '\u{80000}'..='\u{8FFFD}'
        | '\u{90000}'..='\u{9FFFD}'
        | '\u{A0000}'..='\u{AFFFD}'
        | '\u{B0000}'..='\u{BFFFD}'
        | '\u{C0000}'..='\u{CFFFD}'
        | '\u{D0000}'..='\u{DFFFD}'
        | '\u{E1000}'..='\u{EFFFD}'
    )
}

/// [RFC 3987 §2.2] `iprivate`: additionally allowed, unescaped, only inside the query
/// component (this is the mechanism by which IRI schemes such as `x-agent-application`
/// carve out private-use ranges for query parameters).
///
/// ```text
/// iprivate = %xE000-F8FF / %xF0000-FFFFD / %x100000-10FFFD
/// ```
///
/// [RFC 3987 §2.2]: https://www.rfc-editor.org/rfc/rfc3987#section-2.2
#[inline]
#[must_use]
pub(crate) fn is_iprivate(c: char) -> bool {
    matches!(c, '\u{E000}'..='\u{F8FF}' | '\u{F0000}'..='\u{FFFFD}' | '\u{100000}'..='\u{10FFFD}')
}

/// `ALPHA / DIGIT / "-" / "." / "_" / "~" / sub-delims` — every unreserved and sub-delimiter
/// ASCII byte, used as the ASCII fast path for host name validation (a host has no `ipchar`
/// punctuation like `:` or `@`, unlike a path segment or userinfo).
const HOST_ASCII: AsciiSet = {
    let mut set = oxixml_text::percent::UNRESERVED;
    set = set.add(b'!').add(b'$').add(b'&').add(b'\'');
    set = set.add(b'(').add(b')').add(b'*').add(b'+');
    set.add(b',').add(b';').add(b'=')
};

/// `ALPHA / DIGIT / "+" / "-" / "."` — the `scheme` production's allowed bytes.
const SCHEME_ASCII: AsciiSet = {
    let mut set = AsciiSet::empty()
        .add_range(b'A', b'Z')
        .add_range(b'a', b'z');
    set = set.add_range(b'0', b'9');
    set.add(b'+').add(b'-').add(b'.')
};

/// Validates an `ischeme` production: `ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )`.
///
/// # Errors
///
/// [`IriParseErrorKind::EmptyScheme`] if `scheme` is empty, or
/// [`IriParseErrorKind::InvalidSchemeCharacter`] for the first offending character (the first
/// character is additionally required to be `ALPHA`, not just a member of [`SCHEME_ASCII`]).
pub(crate) fn validate_scheme(scheme: &str) -> Result<(), IriParseError> {
    let mut chars = scheme.chars();
    let first = chars.next().ok_or(IriParseErrorKind::EmptyScheme)?;
    if !first.is_ascii_alphabetic() {
        return Err(IriParseErrorKind::InvalidSchemeCharacter(first).into());
    }
    for c in chars {
        if !(c.is_ascii() && SCHEME_ASCII.contains(c as u8)) {
            return Err(IriParseErrorKind::InvalidSchemeCharacter(c).into());
        }
    }
    Ok(())
}

/// Validates an `iauthority` production: `[ iuserinfo "@" ] ihost [ ":" port ]`.
///
/// # Errors
///
/// Returns an error rooted in whichever sub-production (`iuserinfo`, `ihost`, `IP-literal`, or
/// `port`) first rejects a byte or code point.
pub(crate) fn validate_authority(authority: &str) -> Result<(), IriParseError> {
    // Fast path: authorities are almost always a plain ASCII `unreserved`/`sub-delims` host
    // with no userinfo, no port, no brackets, and no percent-encoding — skip straight to
    // success. Deliberately *excludes* `:` from the fast set: a `:` might separate a port
    // (which must still be checked as `*DIGIT`) or a userinfo, so its presence always falls
    // through to the slower, structurally-precise path below.
    if authority.bytes().all(|b| HOST_ASCII.contains(b)) {
        return Ok(());
    }
    const USERINFO_ASCII: AsciiSet = oxixml_text::percent::USERINFO;

    let mut remaining = authority;
    if let Some(at_index) = memchr(b'@', remaining.as_bytes()) {
        let userinfo = &remaining[..at_index];
        remaining = &remaining[at_index + 1..];
        validate_code_point_or_pct(userinfo, USERINFO_ASCII, is_ucschar)?;
    }

    if let Some(after_bracket) = remaining.strip_prefix('[') {
        let Some(close) = memchr(b']', after_bracket.as_bytes()) else {
            return Err(IriParseErrorKind::UnmatchedHostBracket.into());
        };
        validate_ip_literal(&after_bracket[..close])?;
        let rest = &after_bracket[close + 1..];
        if !rest.is_empty() {
            match rest.strip_prefix(':') {
                Some(port) => validate_port(port)?,
                None => {
                    let bad = rest.chars().next().unwrap_or_default();
                    return Err(IriParseErrorKind::InvalidHostCharacter(bad).into());
                }
            }
        }
    } else {
        let host = if let Some(colon_index) = memchr(b':', remaining.as_bytes()) {
            validate_port(&remaining[colon_index + 1..])?;
            &remaining[..colon_index]
        } else {
            remaining
        };
        validate_code_point_or_pct(host, HOST_ASCII, is_ucschar)?;
    }
    Ok(())
}

/// Validates the interior of an `IP-literal`'s brackets: either an `IPv6address` (delegated to
/// [`std::net::Ipv6Addr`]'s parser) or an `IPvFuture`.
fn validate_ip_literal(ip: &str) -> Result<(), IriParseError> {
    if ip.starts_with(['v', 'V']) {
        validate_ip_v_future(ip)
    } else {
        match Ipv6Addr::from_str(ip) {
            Ok(_) => Ok(()),
            Err(e) => Err(IriParseErrorKind::InvalidHostIp(e).into()),
        }
    }
}

/// Validates an `IPvFuture` production: `"v" 1*HEXDIG "." 1*( unreserved / sub-delims / ":" )`.
fn validate_ip_v_future(ip: &str) -> Result<(), IriParseError> {
    const IP_FUTURE_TAIL: AsciiSet = oxixml_text::percent::USERINFO;

    let Some(mut rest) = ip.strip_prefix(['v', 'V']) else {
        return Err(
            IriParseErrorKind::InvalidHostCharacter(ip.chars().next().unwrap_or_default()).into(),
        );
    };
    let version_len = rest.bytes().take_while(u8::is_ascii_hexdigit).count();
    if version_len == 0 {
        return Err(IriParseErrorKind::InvalidHostCharacter(
            rest.chars().next().unwrap_or_default(),
        )
        .into());
    }
    rest = &rest[version_len..];
    let Some(tail) = rest.strip_prefix('.') else {
        return Err(IriParseErrorKind::InvalidHostCharacter(
            rest.chars().next().unwrap_or_default(),
        )
        .into());
    };
    if tail.is_empty() {
        return Err(IriParseErrorKind::InvalidHostCharacter(']').into());
    }
    for c in tail.chars() {
        if !(c.is_ascii() && IP_FUTURE_TAIL.contains(c as u8)) {
            return Err(IriParseErrorKind::InvalidHostCharacter(c).into());
        }
    }
    Ok(())
}

/// Validates a `port` production: `*DIGIT`.
fn validate_port(port: &str) -> Result<(), IriParseError> {
    for c in port.chars() {
        if !c.is_ascii_digit() {
            return Err(IriParseErrorKind::InvalidPortCharacter(c).into());
        }
    }
    Ok(())
}

/// `ipchar / "/"` — the ASCII fast set for an `ipath-*` production.
const PATH_ASCII: AsciiSet = oxixml_text::percent::PATH;
/// `ipchar / "/" / "?"` — the ASCII fast set shared by `iquery` and `ifragment`.
const QUERY_OR_FRAGMENT_ASCII: AsciiSet = oxixml_text::percent::QUERY;

/// Validates an `ipath-*` production (`isegment *( "/" isegment )`, in any of its five
/// path-kind variants — this crate does not distinguish between them since the grammars only
/// differ in whether the *first* segment may contain a `:`, which the caller already fixed by
/// construction of the authority/path split).
pub(crate) fn validate_path(path: &str) -> Result<(), IriParseError> {
    validate_code_point_or_pct(path, PATH_ASCII, is_ucschar)
}

/// Validates an `iquery` production: `*( ipchar / iprivate / "/" / "?" )`.
pub(crate) fn validate_query(query: &str) -> Result<(), IriParseError> {
    validate_code_point_or_pct(query, QUERY_OR_FRAGMENT_ASCII, |c| {
        is_ucschar(c) || is_iprivate(c)
    })
}

/// Validates an `ifragment` production: `*( ipchar / "/" / "?" )` (unlike `iquery`, no
/// `iprivate`).
pub(crate) fn validate_fragment(fragment: &str) -> Result<(), IriParseError> {
    validate_code_point_or_pct(fragment, QUERY_OR_FRAGMENT_ASCII, is_ucschar)
}

/// Walks `input` once, accepting: any byte in `ascii_set`; a well-formed `pct-encoded` triplet
/// (`%` followed by two hex digits — the *decoded* byte is not further validated, matching
/// RFC 3987, which treats `pct-encoded` as opaque); or, for a byte `>= 0x80`, a full [`char`]
/// accepted by `is_extra` (the component-specific `ucschar`/`iprivate` test).
///
/// # Errors
///
/// [`IriParseErrorKind::InvalidPercentEncoding`] for a `%` not followed by two hex digits, or
/// [`IriParseErrorKind::InvalidIriCodePoint`] for any other rejected character.
fn validate_code_point_or_pct(
    input: &str,
    ascii_set: AsciiSet,
    is_extra: impl Fn(char) -> bool,
) -> Result<(), IriParseError> {
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b if b < 0x80 && ascii_set.contains(b) => i += 1,
            b'%' => {
                let valid = bytes.get(i + 1).is_some_and(u8::is_ascii_hexdigit)
                    && bytes.get(i + 2).is_some_and(u8::is_ascii_hexdigit);
                if !valid {
                    let mut chars = input[i + 1..].chars();
                    let c1 = chars.next();
                    let c2 = chars.next();
                    return Err(
                        IriParseErrorKind::InvalidPercentEncoding([Some('%'), c1, c2]).into(),
                    );
                }
                i += 3;
            }
            b if b >= 0x80 => {
                let c = input[i..].chars().next().unwrap_or_default();
                if !is_extra(c) {
                    return Err(IriParseErrorKind::InvalidIriCodePoint(c).into());
                }
                i += c.len_utf8();
            }
            b => {
                return Err(IriParseErrorKind::InvalidIriCodePoint(char::from(b)).into());
            }
        }
    }
    Ok(())
}

/// Validates a full [`Iri`]: requires a `scheme` (an `Iri` is always absolute), then delegates
/// to [`validate_iri_ref`] for the remaining components.
///
/// # Errors
///
/// [`IriParseErrorKind::NoScheme`] if `iri` has no scheme, or any error from
/// [`validate_iri_ref`].
pub(crate) fn validate_iri<T: Deref<Target = str>>(iri: &Iri<T>) -> Result<(), IriParseError> {
    let iri_ref: &IriRef<T> = iri.as_iri_ref();
    if !iri_ref.is_absolute() {
        return Err(IriParseErrorKind::NoScheme.into());
    }
    validate_iri_ref(iri_ref)
}

/// Validates an [`IriRef`] by validating each of its (already delimited) components in turn:
/// `scheme`, `authority`, `path`, `query`, `fragment`.
///
/// # Errors
///
/// The first error raised by any component's `validate_*` function, in component order.
pub(crate) fn validate_iri_ref<T: Deref<Target = str>>(
    iri: &IriRef<T>,
) -> Result<(), IriParseError> {
    if let Some(scheme) = iri.scheme() {
        validate_scheme(scheme)?;
    }
    if let Some(authority) = iri.authority() {
        validate_authority(authority)?;
    }
    validate_path(iri.path())?;
    if let Some(query) = iri.query() {
        validate_query(query)?;
    }
    if let Some(fragment) = iri.fragment() {
        validate_fragment(fragment)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ucschar_covers_the_bmp_and_excludes_gaps_and_private_use() {
        assert!(is_ucschar('\u{A0}')); // range start
        assert!(is_ucschar('\u{D7FF}')); // just below surrogates
        assert!(is_ucschar('\u{4E00}')); // CJK ideograph
        assert!(!is_ucschar('\u{9F}')); // just below range start
        assert!(!is_ucschar('a')); // ASCII is not ucschar (handled separately)
        assert!(!is_ucschar('\u{FFFF}')); // noncharacter, excluded
        assert!(is_ucschar('\u{10000}')); // first supplementary-plane range
        assert!(is_ucschar('\u{EFFFD}')); // last ucschar range, upper edge
        // U+F0000-FFFFD and U+100000-10FFFD are `iprivate`, not `ucschar` (RFC 3987 §2.2).
        assert!(!is_ucschar('\u{F0000}'));
        assert!(!is_ucschar('\u{10FFFD}'));
    }

    #[test]
    fn iprivate_covers_exactly_the_three_private_use_ranges() {
        assert!(is_iprivate('\u{E000}'));
        assert!(is_iprivate('\u{F8FF}'));
        assert!(!is_iprivate('\u{F900}')); // just above, belongs to ucschar instead
        assert!(is_iprivate('\u{F0000}'));
        assert!(is_iprivate('\u{10FFFD}'));
        assert!(!is_iprivate('a'));
    }

    #[test]
    fn scheme_validation_accepts_the_full_production() -> Result<(), IriParseError> {
        validate_scheme("a")?;
        validate_scheme("http")?;
        validate_scheme("x-custom+scheme.v2")?;
        validate_scheme("A1")?;
        Ok(())
    }

    #[test]
    fn scheme_validation_rejects_empty_and_malformed_input() {
        assert!(validate_scheme("").is_err());
        assert!(validate_scheme("1http").is_err());
        assert!(validate_scheme("ht tp").is_err());
        assert!(validate_scheme("-http").is_err());
    }

    #[test]
    fn port_validation_accepts_digits_and_empty_and_rejects_letters() {
        assert!(validate_port("").is_ok());
        assert!(validate_port("8080").is_ok());
        assert!(validate_port("0").is_ok());
        assert!(validate_port("80a").is_err());
    }

    #[test]
    fn ip_v_future_requires_hex_version_dot_and_nonempty_tail() {
        assert!(validate_ip_v_future("v1.a").is_ok());
        assert!(validate_ip_v_future("vAB.some-token:more").is_ok());
        assert!(validate_ip_v_future("v.a").is_err()); // no hex digits before '.'
        assert!(validate_ip_v_future("v1a").is_err()); // no '.'
        assert!(validate_ip_v_future("v1.").is_err()); // empty tail
    }

    #[test]
    fn authority_fast_path_still_rejects_bad_ports() {
        // Every byte of "example.com:xy" is within the colon-free fast-path set once you
        // exclude the ':' itself, so this authority must fall through to the structural
        // (slow) path and be rejected there for its non-digit port, not accepted early.
        assert!(validate_authority("example.com:xy").is_err());
        assert!(validate_authority("example.com:80").is_ok());
        assert!(validate_authority("example.com").is_ok());
    }

    #[test]
    fn authority_validates_bracketed_ipv6_and_rejects_unmatched_brackets() {
        assert!(validate_authority("[::1]").is_ok());
        assert!(validate_authority("[::1]:80").is_ok());
        assert!(validate_authority("[::1").is_err());
        assert!(validate_authority("[gggg::1]").is_err());
    }

    #[test]
    fn percent_encoding_validation_requires_two_hex_digits() {
        assert!(validate_path("/a%20b").is_ok());
        assert!(validate_path("/a%2").is_err());
        assert!(validate_path("/a%").is_err());
        assert!(validate_path("/a%zz").is_err());
    }
}
