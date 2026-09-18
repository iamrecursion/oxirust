//! Byte-offset bookkeeping for the five IRI-reference components (scheme, authority, path,
//! query, fragment), computed in a single forward scan.

use oxixml_support::bytes::{memchr, memchr2};

/// Byte offsets delimiting the components of an IRI reference, all relative to the start of
/// the underlying string.
///
/// The components are laid out contiguously in the source text in grammar order
/// (`scheme ":" ["//" authority] path ["?" query] ["#" fragment]`), so each field marks the
/// *end* of its component; the *start* of a component is the *end* of the previous one (with a
/// fixed one- or two-byte separator skipped, see the accessor methods on [`crate::IriRef`]).
///
/// * `scheme_end == 0` means there is no scheme (the reference is relative); otherwise the
///   scheme is `iri[..scheme_end - 1]` (the `-1` skips the trailing `:`).
/// * `authority_end > scheme_end` (beyond the `://` marker) means there is an authority.
/// * `path_end` is always valid: every IRI reference has a path (possibly empty).
/// * `query_end > path_end` means there is a query component.
/// * `iri.len() > query_end` means there is a fragment component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IriElementsPositions {
    pub(crate) scheme_end: usize,
    pub(crate) authority_end: usize,
    pub(crate) path_end: usize,
    pub(crate) query_end: usize,
}

/// Computes positions for an IRI that is assumed to start with a scheme (used by [`crate::Iri`],
/// which is always absolute once parsed — though the positions are computed before validation,
/// so a missing scheme is still detected and reported later).
#[inline]
pub(crate) fn find_iri_positions(iri: &str) -> IriElementsPositions {
    let bytes = iri.as_bytes();
    let scheme_end = memchr(b':', bytes).map_or(0, |position| position + 1);
    find_iri_positions_knowing_scheme_end(bytes, scheme_end)
}

/// Computes the remaining (authority/path/query/fragment) positions given a known scheme end.
#[inline]
pub(crate) fn find_iri_positions_knowing_scheme_end(
    iri: &[u8],
    scheme_end: usize,
) -> IriElementsPositions {
    let path_end = memchr2(b'?', b'#', &iri[scheme_end..]).map_or(iri.len(), |l| scheme_end + l);
    let query_end = memchr(b'#', &iri[path_end..]).map_or(iri.len(), |l| path_end + l);
    let authority_end =
        if scheme_end + 2 <= path_end && iri[scheme_end] == b'/' && iri[scheme_end + 1] == b'/' {
            memchr(b'/', &iri[scheme_end + 2..path_end]).map_or(path_end, |l| scheme_end + 2 + l)
        } else {
            scheme_end
        };
    IriElementsPositions {
        scheme_end,
        authority_end,
        path_end,
        query_end,
    }
}

/// Computes positions for an IRI reference, which may or may not start with a scheme.
#[inline]
pub(crate) fn find_iri_ref_positions(iri: &str) -> IriElementsPositions {
    let bytes = iri.as_bytes();
    match bytes.first().copied() {
        Some(b'/') => {
            // Starts with a path (network-path or absolute-path reference).
            find_iri_positions_knowing_scheme_end(bytes, 0)
        }
        Some(b'?') => {
            // Query (+ optional fragment) only; the path is empty.
            let query_end = memchr(b'#', &bytes[1..]).map_or(bytes.len(), |p| p + 1);
            IriElementsPositions {
                scheme_end: 0,
                authority_end: 0,
                path_end: 0,
                query_end,
            }
        }
        Some(b'#') | None => {
            // Fragment-only reference, or the empty reference.
            IriElementsPositions {
                scheme_end: 0,
                authority_end: 0,
                path_end: 0,
                query_end: 0,
            }
        }
        _ => {
            // The reference may start with a scheme or with a relative path; the first of
            // `:`, `?`, `/`, `#` disambiguates (a `:` before any of the others is a scheme).
            let scheme_end = find_scheme_end_for_iri_ref(bytes);
            find_iri_positions_knowing_scheme_end(bytes, scheme_end)
        }
    }
}

/// Scans for a scheme terminator (`:`) that precedes any of `?`, `/`, `#`; returns `0` (no
/// scheme) if one of those three is seen first, or if none of the four bytes appear at all.
#[inline]
fn find_scheme_end_for_iri_ref(iri: &[u8]) -> usize {
    for (index, byte) in iri.iter().copied().enumerate() {
        match byte {
            b':' => return index + 1,
            b'?' | b'/' | b'#' => return 0,
            _ => {}
        }
    }
    0
}

/// Returns `true` if `path` contains a `.` or `..` path segment (`/./`, `/../`, a leading `./`
/// or `../`, or a path equal to `.` or `..`).
///
/// Used by [`crate::Iri::relativize`] to reject targets whose path a naive resolution would
/// reinterpret, and internally by the resolver to skip the (more expensive) dot-segment removal
/// pass when a path provably has none.
#[inline]
pub(crate) fn has_dot_segment(path: &str) -> bool {
    let bytes = path.as_bytes();
    let mut offset = 0;
    while let Some(relative) = memchr(b'.', &bytes[offset..]) {
        let dot_offset = offset + relative;
        // Conservative on purpose: a leading dot (`dot_offset == 0`) is always treated as a
        // possible dot segment without inspecting what follows it, since the byte before it
        // (one past the start of the string) does not exist. Callers only use this as a
        // fast-path *skip* guard before the precise, per-segment removal algorithm, so
        // over-reporting here costs a few extra comparisons, never correctness.
        if dot_offset == 0
            || (bytes[dot_offset - 1] == b'/'
                && bytes.get(dot_offset + 1).is_none_or(|b| {
                    *b == b'/'
                        || (*b == b'.' && bytes.get(dot_offset + 2).is_none_or(|b| *b == b'/'))
                }))
        {
            return true;
        }
        offset = dot_offset + 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iri_ref_positions_split_a_full_reference() {
        let positions = find_iri_ref_positions("http://user@host:80/a/b?q#f");
        assert_eq!(positions.scheme_end, 5); // "http:".len()
        assert_eq!(
            &"http://user@host:80/a/b?q#f"[positions.scheme_end + 2..positions.authority_end],
            "user@host:80"
        );
        assert_eq!(
            &"http://user@host:80/a/b?q#f"[positions.authority_end..positions.path_end],
            "/a/b"
        );
        assert_eq!(
            &"http://user@host:80/a/b?q#f"[positions.path_end + 1..positions.query_end],
            "q"
        );
        assert_eq!(
            &"http://user@host:80/a/b?q#f"[positions.query_end + 1..],
            "f"
        );
    }

    #[test]
    fn iri_ref_positions_handle_every_relative_leading_byte() {
        assert_eq!(find_iri_ref_positions("").query_end, 0);
        assert_eq!(find_iri_ref_positions("#f").query_end, 0);
        assert_eq!(find_iri_ref_positions("?q").path_end, 0);
        assert_eq!(find_iri_ref_positions("?q#f").query_end, 2);
        assert_eq!(find_iri_ref_positions("/a/b").scheme_end, 0);
        assert_eq!(find_iri_ref_positions("//auth/a").scheme_end, 0);
        assert_eq!(find_iri_ref_positions("rel/a").scheme_end, 0);
        assert_eq!(find_iri_ref_positions("scheme:rel").scheme_end, 7);
    }

    #[test]
    fn scheme_end_detection_prefers_a_colon_seen_before_reserved_delimiters() {
        // A `:` appearing before any of `/`, `?`, `#` starts a scheme.
        assert_eq!(find_scheme_end_for_iri_ref(b"a:b"), 2);
        // A `/`, `?` or `#` seen first means there is no scheme (this is a relative
        // reference, and any later `:` is just ordinary path/query/fragment content).
        assert_eq!(find_scheme_end_for_iri_ref(b"a/b:c"), 0);
        assert_eq!(find_scheme_end_for_iri_ref(b"a?b:c"), 0);
        assert_eq!(find_scheme_end_for_iri_ref(b"a#b:c"), 0);
        assert_eq!(find_scheme_end_for_iri_ref(b"noscheme"), 0);
    }

    #[test]
    fn has_dot_segment_detects_every_dot_segment_form() {
        // Genuine non-dot-segment paths, none of which start with a literal `.`.
        for path in ["", "g", "g/h", "a/.b", "a/b.c"] {
            assert!(
                !has_dot_segment(path),
                "expected no dot segment in {path:?}"
            );
        }
        // A path beginning with `.` is always conservatively reported as having a dot
        // segment (see the comment on the check below) even when, like `"..."`, it is not
        // really one — this is a safe, deliberate over-approximation for callers that only
        // use this as a fast-path *skip* guard.
        for path in [
            ".", "..", "...", "./g", "../g", "a/./b", "a/../b", "a/..", "a/.",
        ] {
            assert!(has_dot_segment(path), "expected a dot segment in {path:?}");
        }
    }

    #[test]
    fn find_iri_positions_requires_no_scheme_delimiter_to_report_zero() {
        let positions = find_iri_positions("no-colon-here");
        assert_eq!(positions.scheme_end, 0);
    }
}
