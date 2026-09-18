//! Normalization rules from OASIS XML Catalogs 1.1, §6 ("Comparing URI
//! References") and §6.5 ("Comparing Public Identifiers").
//!
//! Catalog lookups never compare raw strings: every system identifier, URI
//! reference and public identifier is normalized first, so that
//! percent-encoding and incidental whitespace differences do not defeat a
//! match. Both normalization functions are idempotent, matching the spec's
//! requirement that re-normalizing an already-normalized value is a no-op.

/// Normalizes a URI reference (a system identifier, a `uri`/`name` attribute
/// value, or a `*StartString`/`*Suffix` matching string) per the percent-encoding
/// rule in OASIS XML Catalogs 1.1 §6.
///
/// Every character that is non-ASCII, or that falls in the "excluded"
/// character set of [RFC 2396 §2.4](https://www.rfc-editor.org/rfc/rfc2396#section-2.4)
/// (control characters, space, and the delimiters/unwise characters `< > " { }
/// | \ ^` `` ` ``), is replaced by the `%HH` percent-encoding of its UTF-8
/// bytes. The number sign (`#`), percent sign (`%`), and square brackets (`[`
/// `]`) are explicitly exempted so that fragments, existing escapes, and
/// [RFC 2732](https://www.rfc-editor.org/rfc/rfc2732) `IPv6reference` hosts
/// survive normalization unchanged.
#[must_use]
pub(crate) fn normalize_uri_ref(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if is_disallowed(ch) {
            let mut buf = [0u8; 4];
            for byte in ch.encode_utf8(&mut buf).as_bytes() {
                out.push('%');
                out.push(hex_digit(byte >> 4));
                out.push(hex_digit(byte & 0x0f));
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Returns `true` for characters that must be percent-encoded by
/// [`normalize_uri_ref`]: every non-ASCII character, plus the ASCII control
/// characters, space, and the RFC 2396 "excluded" delimiter/unwise characters —
/// except for `#`, `%`, `[` and `]`, which are always left as-is.
const fn is_disallowed(ch: char) -> bool {
    if !ch.is_ascii() {
        return true;
    }
    if matches!(ch, '#' | '%' | '[' | ']') {
        return false;
    }
    let byte = ch as u32;
    byte <= 0x20
        || byte == 0x7f
        || matches!(ch, '<' | '>' | '"' | '{' | '}' | '|' | '\\' | '^' | '`')
}

const fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'A' + (nibble - 10)) as char,
    }
}

/// The scheme prefix an RFC 3151 "publicid" URN carries, matched
/// case-insensitively as RFC 8141 requires of URN schemes.
const PUBLICID_URN_PREFIX: &str = "urn:publicid:";

/// "Unwraps" an RFC 3151 `urn:publicid:` URN back into the public identifier it
/// encodes, per OASIS XML Catalogs 1.1 §6.4; returns [`None`] when `value` is
/// not such a URN.
///
/// The transformation is the inverse of RFC 3151 §3: the URN-specific string
/// has `+` turned back into a space, `:` into `//` and `;` into `::`, and then
/// every `%HH` escape is decoded (which is why the substitutions run first —
/// an *escaped* `+`, `:` or `;` is still `%2B`/`%3A`/`%3B` at that point and is
/// therefore left alone). The result is normalized with
/// [`normalize_public_id`], so it can be compared directly against the
/// `publicId` of a `public` entry.
#[must_use]
pub(crate) fn unwrap_publicid(value: &str) -> Option<String> {
    let trimmed = value.trim();
    // `get` returns `None` both when `trimmed` is shorter than the prefix and
    // when the prefix's length lands inside a multi-byte character, so this
    // can never panic the way a bare `split_at` would on arbitrary input.
    let scheme = trimmed.get(..PUBLICID_URN_PREFIX.len())?;
    if !scheme.eq_ignore_ascii_case(PUBLICID_URN_PREFIX) {
        return None;
    }
    // Safe: `scheme` above only exists when `PUBLICID_URN_PREFIX.len()` is a
    // valid char boundary of `trimmed`.
    let rest = &trimmed[PUBLICID_URN_PREFIX.len()..];
    let mut substituted = String::with_capacity(rest.len());
    for ch in rest.chars() {
        match ch {
            '+' => substituted.push(' '),
            ':' => substituted.push_str("//"),
            ';' => substituted.push_str("::"),
            other => substituted.push(other),
        }
    }
    Some(normalize_public_id(&percent_decode(&substituted)))
}

/// Decodes every `%HH` escape of `value` as a UTF-8 byte sequence, leaving a
/// malformed escape (or one that does not form valid UTF-8) verbatim.
fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let (Some(high), Some(low)) = (
                (bytes[index + 1] as char).to_digit(16),
                (bytes[index + 2] as char).to_digit(16),
            )
        {
            #[allow(clippy::cast_possible_truncation)]
            out.push((high * 16 + low) as u8);
            index += 3;
            continue;
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| value.to_owned())
}

/// Normalizes a public identifier per OASIS XML Catalogs 1.1 §6.5: every run of
/// whitespace is collapsed to a single space (`#x20`), and leading/trailing
/// whitespace is removed.
#[must_use]
pub(crate) fn normalize_public_id(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut in_space = false;
    for ch in value.chars() {
        if ch.is_whitespace() {
            in_space = true;
        } else {
            if in_space && !out.is_empty() {
                out.push(' ');
            }
            in_space = false;
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{normalize_public_id, normalize_uri_ref, unwrap_publicid};

    #[test]
    fn unwrap_publicid_does_not_panic_on_a_multibyte_char_straddling_the_prefix_boundary() {
        // 12 ASCII bytes of the prefix (missing its trailing `:`) immediately
        // followed by `\u{20AC}` (`€`, 3 bytes in UTF-8, occupying bytes
        // 12..15) — byte offset 13 (`PUBLICID_URN_PREFIX.len()`) lands inside
        // that character rather than on a boundary.
        assert_eq!(
            unwrap_publicid("urn:publicid\u{20ac}rest-of-the-system-id"),
            None
        );
        // A handful of other multi-byte characters at, before, and after the
        // same offset, so the boundary check is exercised on both sides.
        assert_eq!(unwrap_publicid("urn:publici\u{20ac}d:rest"), None);
        assert_eq!(unwrap_publicid("urn:public\u{20ac}id:rest"), None);
        assert_eq!(unwrap_publicid("\u{20ac}"), None);
        assert_eq!(unwrap_publicid(""), None);
        // The exact-length ASCII prefix with nothing after it still unwraps.
        assert_eq!(unwrap_publicid("urn:publicid:"), Some(String::new()));
    }

    #[test]
    fn uri_ref_escapes_space_and_leaves_hash_percent() {
        assert_eq!(normalize_uri_ref("a b"), "a%20b");
        assert_eq!(normalize_uri_ref("a#b"), "a#b");
        assert_eq!(normalize_uri_ref("a%20b"), "a%20b");
        assert_eq!(normalize_uri_ref("[::1]"), "[::1]");
    }

    #[test]
    fn uri_ref_escapes_non_ascii() {
        assert_eq!(normalize_uri_ref("caf\u{e9}"), "caf%C3%A9");
    }

    #[test]
    fn uri_ref_is_idempotent() {
        let once = normalize_uri_ref("a b#c%20[::1]");
        let twice = normalize_uri_ref(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn public_id_collapses_whitespace() {
        assert_eq!(
            normalize_public_id("  -//OASIS//DTD  DocBook   XML//EN "),
            "-//OASIS//DTD DocBook XML//EN"
        );
    }
}
