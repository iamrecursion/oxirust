//! [RFC 3986 §5] relative resolution: turning a base IRI and a (possibly relative) IRI
//! reference into an absolute-form target IRI reference, using the strict (non
//! backward-compatible) algorithm of §5.2 together with the `remove_dot_segments` merge of
//! §5.2.4.

use oxixml_support::bytes::memrchr;

use crate::error::IriParseErrorKind;
use crate::iri_ref::IriRef;
use crate::positions::{IriElementsPositions, has_dot_segment};
use std::ops::Deref;

/// The output sink used by [`crate::IriRef::resolve_into`] and [`crate::Iri::resolve_into`] to
/// build the resolved IRI without an extra intermediate allocation when the caller supplies a
/// reusable buffer.
///
/// Implemented for [`String`]; every method has the same semantics as the identically named
/// [`String`] method.
pub trait OutputBuffer {
    /// Returns the buffer's current contents.
    fn as_str(&self) -> &str;
    /// Appends `s` to the end of the buffer.
    fn push_str(&mut self, s: &str);
    /// Reserves capacity for at least `additional` more bytes, without over-allocating.
    fn reserve_exact(&mut self, additional: usize);
    /// Shortens the buffer to `len` bytes. `len` must be a valid UTF-8 boundary.
    fn truncate(&mut self, len: usize);
}

impl OutputBuffer for String {
    #[inline]
    fn as_str(&self) -> &str {
        self.as_str()
    }

    #[inline]
    fn push_str(&mut self, s: &str) {
        self.push_str(s);
    }

    #[inline]
    fn reserve_exact(&mut self, additional: usize) {
        self.reserve_exact(additional);
    }

    #[inline]
    fn truncate(&mut self, len: usize) {
        self.truncate(len);
    }
}

/// Implements the [RFC 3986 §5.3] "Component Recomposition" transform, dispatching on which of
/// `relative`'s components are present exactly as the pseudocode in §5.3 does, and writes the
/// result into `output_buffer`. Returns the resolved positions and, if the strict algorithm
/// detected a case it refuses to handle (see the two error kinds below), the error to report.
///
/// [RFC 3986 §5.3]: https://www.rfc-editor.org/rfc/rfc3986#section-5.3
#[inline]
pub(crate) fn resolve<T1: Deref<Target = str>, T2: Deref<Target = str>>(
    base: &IriRef<T1>,
    relative: &IriRef<T2>,
    output_buffer: &mut impl OutputBuffer,
) -> (IriElementsPositions, Option<IriParseErrorKind>) {
    let mut error = None;
    let positions = if relative.is_absolute() {
        output_buffer.reserve_exact(relative.iri.len());
        output_buffer.push_str(&relative.iri[..relative.positions.authority_end]);
        write_path_without_dot_segments_to(
            relative.path(),
            output_buffer,
            relative.positions.authority_end,
            false,
        );
        let path_end = output_buffer.as_str().len();
        output_buffer.push_str(&relative.iri[relative.positions.path_end..]);
        IriElementsPositions {
            scheme_end: relative.positions.scheme_end,
            authority_end: relative.positions.authority_end,
            path_end,
            query_end: path_end + (relative.positions.query_end - relative.positions.path_end),
        }
    } else if relative.positions.authority_end > 0 {
        output_buffer.reserve_exact(base.positions.scheme_end + relative.iri.len());
        output_buffer.push_str(&base.iri[..base.positions.scheme_end]);
        output_buffer.push_str(&relative.iri[..relative.positions.authority_end]);
        write_path_without_dot_segments_to(
            relative.path(),
            output_buffer,
            base.positions.scheme_end + relative.positions.authority_end,
            false,
        );
        let path_end = output_buffer.as_str().len();
        output_buffer.push_str(&relative.iri[relative.positions.path_end..]);
        IriElementsPositions {
            scheme_end: base.positions.scheme_end,
            authority_end: base.positions.scheme_end + relative.positions.authority_end,
            path_end,
            query_end: path_end + (relative.positions.query_end - relative.positions.path_end),
        }
    } else if relative.positions.path_end > 0 {
        if relative.iri.starts_with('/') {
            output_buffer.reserve_exact(base.positions.authority_end + relative.iri.len());
            output_buffer.push_str(&base.iri[..base.positions.authority_end]);
            write_path_without_dot_segments_to(
                &relative.iri[..relative.positions.path_end],
                output_buffer,
                base.positions.authority_end,
                false,
            );
        } else if base.positions.authority_end > base.positions.scheme_end
            && base.positions.authority_end == base.positions.path_end
        {
            output_buffer.reserve_exact(
                base.positions.authority_end
                    + 1
                    + (relative.iri.len() - relative.positions.authority_end),
            );
            output_buffer.push_str(&base.iri[..base.positions.authority_end]);
            write_path_without_dot_segments_to(
                relative.path(),
                output_buffer,
                base.positions.authority_end,
                true,
            );
        } else if let Some(last_slash_position) = memrchr(b'/', base.path().as_bytes()) {
            if base.positions.scheme_end == 0 && &base.path()[last_slash_position + 1..] == ".." {
                // The strict algorithm does not support ".." resolving against a base that is
                // itself a bare relative path (no scheme): there is nothing to merge it against.
                error = Some(IriParseErrorKind::ResolvingDotSegmentsNonHierarchical);
            }
            output_buffer.reserve_exact(
                base.positions.authority_end
                    + last_slash_position
                    + (relative.iri.len() - relative.positions.authority_end)
                    + 1,
            );
            output_buffer.push_str(&base.iri[..base.positions.authority_end]);
            write_path_without_dot_segments_to(
                &base.path()[..last_slash_position + 1],
                output_buffer,
                base.positions.authority_end,
                false,
            );
            let with_prefix_slash = if output_buffer.as_str().ends_with('/') {
                output_buffer.truncate(output_buffer.as_str().len() - 1);
                true
            } else {
                false
            };
            write_path_without_dot_segments_to(
                relative.path(),
                output_buffer,
                base.positions.authority_end,
                with_prefix_slash,
            );
        } else {
            if base.positions.authority_end == 0
                && (relative.path().split('/').any(|segment| segment == "..")
                    || base.path() == "..")
            {
                error = Some(IriParseErrorKind::ResolvingDotSegmentsNonHierarchical);
            }
            output_buffer.reserve_exact(
                base.positions.authority_end
                    + (relative.iri.len() - relative.positions.authority_end),
            );
            output_buffer.push_str(&base.iri[..base.positions.authority_end]);
            write_path_without_dot_segments_to(
                relative.path(),
                output_buffer,
                base.positions.authority_end,
                false,
            );
        }
        let path_end = output_buffer.as_str().len();
        output_buffer.push_str(&relative.iri[relative.positions.path_end..]);
        IriElementsPositions {
            scheme_end: base.positions.scheme_end,
            authority_end: base.positions.authority_end,
            path_end,
            query_end: path_end + (relative.positions.query_end - relative.positions.path_end),
        }
    } else if relative.positions.query_end > 0 {
        output_buffer.reserve_exact(base.positions.path_end + relative.iri.len());
        output_buffer.push_str(&base.iri[..base.positions.authority_end]);
        write_path_without_dot_segments_to(
            base.path(),
            output_buffer,
            base.positions.authority_end,
            false,
        );
        let path_end = output_buffer.as_str().len();
        output_buffer.push_str(&relative.iri);
        IriElementsPositions {
            scheme_end: base.positions.scheme_end,
            authority_end: base.positions.authority_end,
            path_end,
            query_end: path_end + relative.positions.query_end,
        }
    } else {
        output_buffer.reserve_exact(base.positions.query_end + relative.iri.len());
        output_buffer.push_str(&base.iri[..base.positions.authority_end]);
        write_path_without_dot_segments_to(
            base.path(),
            output_buffer,
            base.positions.authority_end,
            false,
        );
        let path_end = output_buffer.as_str().len();
        output_buffer.push_str(&base.iri[base.positions.path_end..base.positions.query_end]);
        output_buffer.push_str(&relative.iri);
        IriElementsPositions {
            scheme_end: base.positions.scheme_end,
            authority_end: base.positions.authority_end,
            path_end,
            query_end: path_end + (base.positions.query_end - base.positions.path_end),
        }
    };
    // A relative reference can never legally resolve into something that *looks* like it grew
    // an authority out of nowhere (a path starting with `//` right after the scheme would be
    // reparsed by any later consumer as `scheme://authority...`) — reject it.
    if positions.scheme_end == positions.authority_end
        && output_buffer.as_str()[positions.authority_end..].starts_with("//")
    {
        error = Some(IriParseErrorKind::PathStartingWithTwoSlashes);
    }
    (positions, error)
}

/// Implements [RFC 3986 §5.2.4] `remove_dot_segments`, fused with the "merge" step of §5.2.3:
/// appends `input`'s path (optionally prefixed with a `/` when `with_prefix_slash` is set, used
/// by the merge step when the base has an authority but an empty path) to `output`, resolving
/// any `.`/`..` segments against the portion of `output` written since `output_path_start`.
///
/// When `input` provably contains no dot segment ([`has_dot_segment`]) and the destination is
/// already a well-formed rooted path (or is about to become one), `input` is copied through
/// unmodified — a fast path, and *required* when resolving against a base that is itself a bare
/// relative path, since `remove_dot_segments` would not be meaningful there.
///
/// [RFC 3986 §5.2.4]: https://www.rfc-editor.org/rfc/rfc3986#section-5.2.4
#[inline]
fn write_path_without_dot_segments_to(
    mut input: &str,
    output: &mut impl OutputBuffer,
    output_path_start: usize,
    with_prefix_slash: bool,
) {
    // Whether the buffer written so far already begins with `/` (a rooted path) — if the
    // buffer is still empty, a rooted path is instead predicted from what is about to be
    // written (`with_prefix_slash`, or `input` itself starting with `/`).
    let starts_rooted = match output.as_str().as_bytes().first() {
        Some(b'/') => true,
        Some(_) => false,
        None => with_prefix_slash || input.starts_with('/'),
    };
    if (output_path_start == 0 && !starts_rooted) || !has_dot_segment(input) {
        if with_prefix_slash {
            output.push_str("/");
        }
        output.push_str(input);
        return;
    }
    if with_prefix_slash {
        if input.starts_with("./") {
            // Strip only the leading `.`, deliberately keeping the `/`: the merge step
            // treats this call's `input` as *following* an implicit `/` (that is what
            // `with_prefix_slash` means), so the retained `/` is what re-roots the `while`
            // loop below onto `output_path_start` on its very next iteration.
            input = &input[1..];
        } else if input == "." {
            input = "/";
        } else if input.starts_with("../") {
            input = &input[2..];
            remove_last_segment(output, output_path_start);
        } else if input == ".." {
            input = "/";
            remove_last_segment(output, output_path_start);
        } else {
            output.push_str("/");
            let slash = input.find('/').unwrap_or(input.len());
            output.push_str(&input[..slash]);
            input = &input[slash..];
        }
    }
    while !input.is_empty() {
        if let Some(rest) = input.strip_prefix("../") {
            input = rest;
        } else if let Some(rest) = input.strip_prefix("./") {
            input = rest;
        } else if input.starts_with("/./") {
            input = &input[2..];
        } else if input == "/." {
            input = "/";
        } else if input.starts_with("/../") {
            input = &input[3..];
            remove_last_segment(output, output_path_start);
        } else if input == "/.." {
            input = "/";
            remove_last_segment(output, output_path_start);
        } else if input == "." || input == ".." {
            input = "";
        } else {
            input = if let Some(rest) = input.strip_prefix('/') {
                output.push_str("/");
                rest
            } else {
                input
            };
            let slash = input.find('/').unwrap_or(input.len());
            output.push_str(&input[..slash]);
            input = &input[slash..];
        }
    }
}

/// Truncates `output` back to just before its last `/`-delimited path segment (used when a
/// `..` segment is encountered, to discard the segment it cancels out); never truncates past
/// `output_path_start` (the start of the path within `output`), since `..` above the path root
/// has nothing left to remove.
#[inline]
fn remove_last_segment(output: &mut impl OutputBuffer, output_path_start: usize) {
    let last_slash = memrchr(b'/', &output.as_str().as_bytes()[output_path_start..]).unwrap_or(0);
    output.truncate(output_path_start + last_slash);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs [`write_path_without_dot_segments_to`] against a fresh buffer that already
    /// contains `prefix`, treating the whole of `prefix` as the fixed ("authority") part —
    /// i.e. `output_path_start == prefix.len()`, so nothing before `input` can be climbed
    /// past. Returns the buffer's final contents.
    fn merge(prefix: &str, input: &str, with_prefix_slash: bool) -> String {
        merge_at(prefix, prefix.len(), input, with_prefix_slash)
    }

    /// Variant of [`merge`] with an explicit `output_path_start`, for simulating the second
    /// of the two [`write_path_without_dot_segments_to`] calls `resolve` makes when merging
    /// against a base with a non-empty path: by that point `prefix` already contains *part*
    /// of the base's path (up to its last `/`), so `output_path_start` (the true authority
    /// end) is strictly less than `prefix.len()`.
    fn merge_at(
        prefix: &str,
        output_path_start: usize,
        input: &str,
        with_prefix_slash: bool,
    ) -> String {
        let mut output = prefix.to_owned();
        write_path_without_dot_segments_to(
            input,
            &mut output,
            output_path_start,
            with_prefix_slash,
        );
        output
    }

    #[test]
    fn dot_segments_are_removed_within_a_rooted_path() {
        assert_eq!(merge("http://a", "/b/c/../d", false), "http://a/b/d");
        assert_eq!(merge("http://a", "/b/./c", false), "http://a/b/c");
        assert_eq!(merge("http://a", "/b/c/.", false), "http://a/b/c/");
        assert_eq!(merge("http://a", "/b/c/..", false), "http://a/b/");
    }

    #[test]
    fn merge_with_prefix_slash_reroots_a_leading_climb() {
        // Mirrors the "non-empty base path, merge onto its directory" case (RFC 3986 §5.3):
        // `output_path_start` (8, `"http://a".len()`) is the true authority end, while
        // `prefix` ("http://a/dir") already holds the base path trimmed to its last `/` and
        // then had that trailing `/` cut (which is exactly what `resolve` does before this
        // call, passing the resulting `with_prefix_slash = true`) — as if resolving against
        // a base whose path is `/dir/file`.
        //
        // A leading ".." climbs all the way back to `output_path_start`, discarding "dir"
        // (equivalent to merging "../other" onto "/dir/" -> "/dir/../other" ->  "/other").
        // A leading "." does *not* climb — it names the same directory ("dir" is kept),
        // exactly as `remove_dot_segments("/dir/./other")` yields "/dir/other".
        assert_eq!(
            merge_at("http://a/dir", 8, "../other", true),
            "http://a/other"
        );
        assert_eq!(
            merge_at("http://a/dir", 8, "./other", true),
            "http://a/dir/other"
        );
        assert_eq!(merge_at("http://a/dir", 8, "..", true), "http://a/");
        assert_eq!(merge_at("http://a/dir", 8, ".", true), "http://a/dir/");
    }

    #[test]
    fn merge_with_prefix_slash_onto_a_truly_empty_base_path() {
        // The other `with_prefix_slash = true` call site: a base with an authority but an
        // *entirely* empty path (e.g. resolving "bar" against "http://a"), where
        // `output_path_start == prefix.len()` because nothing has been written yet.
        assert_eq!(merge("http://a", "bar", true), "http://a/bar");
        assert_eq!(merge("http://a", "../bar", true), "http://a/bar");
    }

    #[test]
    fn a_path_with_no_dot_segment_is_copied_through_unmodified() {
        // Fast path: no dot segment anywhere, so the exact bytes of `input` reach `output`.
        assert_eq!(merge("http://a", "/b/c/d", false), "http://a/b/c/d");
        assert_eq!(merge("http://a", "/b/c.txt", false), "http://a/b/c.txt");
    }

    #[test]
    fn bare_relative_base_path_is_not_dot_processed() {
        // `output_path_start == 0` and the destination has no leading '/' (a bare relative
        // path being resolved against another bare relative path): dot removal does not
        // apply, so a literal ".." is passed through untouched rather than climbing.
        assert_eq!(merge("", "../g", false), "../g");
    }

    #[test]
    fn output_buffer_string_impl_matches_string_semantics() {
        let mut buffer = String::from("hello world");
        assert_eq!(OutputBuffer::as_str(&buffer), "hello world");
        OutputBuffer::push_str(&mut buffer, "!");
        assert_eq!(buffer, "hello world!");
        OutputBuffer::truncate(&mut buffer, 5);
        assert_eq!(buffer, "hello");
        OutputBuffer::reserve_exact(&mut buffer, 100);
        assert!(buffer.capacity() >= 105);
    }

    #[test]
    fn remove_last_segment_never_truncates_past_the_path_start() {
        let mut output = "http://a".to_owned();
        let output_path_start = output.len();
        // No '/' at all within the path portion: nothing to remove, stays put.
        remove_last_segment(&mut output, output_path_start);
        assert_eq!(output, "http://a");
    }
}
